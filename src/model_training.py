"""
model_training.py — Model Training Layer

Splits the cleaned data, trains every model in the config-driven registry as
a baseline, tunes each tunable model with the strategy config.yaml assigns
it (GridSearchCV for the models cheap enough to search exhaustively on the
full training fold; a subsample-then-refit HalvingRandomSearchCV for the
expensive tree ensembles), ranks everything by the configured metric, and
evaluates the single best model on the held-out test set exactly once.

This is a classification problem (EDA Section 1: binary Survive target,
~2.1:1 class imbalance) so this module implements the classification branch
only - stratified split, class_weight="balanced" where a model supports it,
Accuracy/Precision/Recall/F1/ROC AUC/PR AUC, ranked by PR AUC (the correct
ranking metric for an imbalanced problem - EDA Section 15/16).

Every fitted pipeline is checkpointed to a JSON ledger + pickle file, keyed
"<stage>::<model name>" (stage in {"baseline", "tuned"}) so the two training
stages can share one ledger without colliding on a bare model name (e.g.
both stages tune a "Random Forest").
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 - required to unlock HalvingRandomSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)

_TYPE_TO_CLASS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "ridge_classifier": RidgeClassifier,
    "lasso_logistic": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


# ==============================================================================
# Split (EDA Section 12: stratified 70/15/15)
# ==============================================================================
def split_data(df: pd.DataFrame, config: dict):
    """Stratified train/validation/test split. Returns
    (X_train, X_val, X_test, y_train, y_val, y_test). The feature matrix
    excludes the target and the ID column (EDA Section 2/10/12: ID is not a
    unique key and carries no clinical signal)."""
    split_cfg = config["split"]
    target = config["data"]["target"]
    id_col = config["data"]["id_column"]

    drop_cols = [target] + ([id_col] if id_col in df.columns else [])
    X = df.drop(columns=drop_cols)
    y = df[target]

    stratify = y if split_cfg["stratify"] else None
    val_test_size = split_cfg["val_size"] + split_cfg["test_size"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_size, stratify=stratify, random_state=split_cfg["random_state"]
    )

    rel_test_size = split_cfg["test_size"] / val_test_size
    stratify_temp = y_temp if split_cfg["stratify"] else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test_size, stratify=stratify_temp, random_state=split_cfg["random_state"]
    )

    logger.info(
        "Split sizes: train=%d (%.1f%%, survive=%.4f)  val=%d (%.1f%%, survive=%.4f)  test=%d (%.1f%%, survive=%.4f)",
        len(X_train), len(X_train) / len(df) * 100, y_train.mean(),
        len(X_val), len(X_val) / len(df) * 100, y_val.mean(),
        len(X_test), len(X_test) / len(df) * 100, y_test.mean(),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==============================================================================
# Model registry
# ==============================================================================
def _build_estimator(model_cfg: dict):
    """Instantiate the sklearn estimator for one registry entry, applying
    base_params and class_weight="balanced" where the model config says to."""
    cls = _TYPE_TO_CLASS[model_cfg["type"]]
    params = dict(model_cfg.get("base_params", {}))
    class_weight = model_cfg.get("class_weight")
    if class_weight:
        params["class_weight"] = class_weight
    return cls(**params)


def build_model_registry(config: dict) -> dict:
    """Build {name: {"estimator": ..., "family":..., "tuning_method":...,
    "uses_sample_weight": bool, "param_grid"/"param_distributions": ...}}
    from config['models'], including the (non-tunable) naive benchmark."""
    registry = {}

    naive_cfg = config["models"]["naive_benchmark"]
    registry[naive_cfg["name"]] = {
        "estimator": DummyClassifier(**naive_cfg["params"]),
        "family": "naive",
        "tuning_method": None,
        "uses_sample_weight": False,
        "tunable": False,
    }

    for model_cfg in config["models"]["registry"]:
        registry[model_cfg["name"]] = {
            "estimator": _build_estimator(model_cfg),
            "family": model_cfg["family"],
            "tuning_method": model_cfg["tuning_method"],
            "uses_sample_weight": model_cfg.get("uses_sample_weight", False),
            "param_grid": model_cfg.get("param_grid"),
            "param_distributions": model_cfg.get("param_distributions"),
            "tunable": True,
        }
    return registry


# ==============================================================================
# Shared evaluation helpers
# ==============================================================================
def evaluate_pipeline(pipeline: Pipeline, X_ev, y_ev) -> dict:
    """Score an already-fitted pipeline. Used identically for baseline and
    tuned pipelines - no per-model special-casing beyond the
    predict_proba/decision_function fallback every model needs, so nothing
    introduced specifically for tuning can silently change what a baseline
    number means."""
    pred = pipeline.predict(X_ev)
    model_step = pipeline.named_steps["model"]
    if hasattr(model_step, "predict_proba"):
        score = pipeline.predict_proba(X_ev)[:, 1]
    else:
        score = pipeline.decision_function(X_ev)
    return {
        "accuracy": float(accuracy_score(y_ev, pred)),
        "precision": float(precision_score(y_ev, pred, zero_division=0)),
        "recall": float(recall_score(y_ev, pred, zero_division=0)),
        "f1": float(f1_score(y_ev, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_ev, score)),
        "pr_auc": float(average_precision_score(y_ev, score)),
    }


def fit_and_evaluate(pipeline: Pipeline, X_tr, y_tr, X_ev, y_ev, fit_kwargs: dict | None = None):
    """Shared fit-then-score helper. fit_kwargs (e.g. sample_weight) is
    always supplied by the CALLER - this function makes no model-specific
    decisions, so a tuning-only adjustment can never leak into a baseline
    fit through this shared code path."""
    pipeline.fit(X_tr, y_tr, **(fit_kwargs or {}))
    return pipeline, evaluate_pipeline(pipeline, X_ev, y_ev)


# ==============================================================================
# Checkpoint ledger (stage::model_name -> metrics, with a pickled pipeline)
# ==============================================================================
class CheckpointLedger:
    """Resumable checkpoint store. Keys are "<stage>::<model_name>" so
    multiple training stages sharing one ledger never collide on a bare
    model name (both "baseline" and "tuned" stages include e.g.
    "Random Forest")."""

    def __init__(self, checkpoints_dir: str | Path):
        self.dir = Path(checkpoints_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.dir / "ledger.json"

    @staticmethod
    def _sanitize(key: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)

    def load(self) -> dict:
        if self.ledger_path.exists():
            with open(self.ledger_path) as f:
                return json.load(f)
        return {}

    def save(self, ledger: dict) -> None:
        with open(self.ledger_path, "w") as f:
            json.dump(ledger, f, indent=2, default=str)

    def checkpoint_or_run(self, stage: str, name: str, compute_fn, force: bool = False):
        """compute_fn() -> (fitted_pipeline, info_dict). If stage::name is
        already checkpointed (ledger entry + pickle both present) and
        force=False, loads and returns that instead of recomputing - this
        is what lets an interrupted run of main.py resume without redoing
        finished work."""
        key = f"{stage}::{name}"
        ledger = self.load()
        ckpt_file = self.dir / f"{self._sanitize(key)}.pkl"

        if not force and key in ledger and ckpt_file.exists():
            with open(ckpt_file, "rb") as f:
                pipeline = pickle.load(f)
            logger.info("[checkpoint] %s: loaded from ledger, skipping recomputation.", key)
            return pipeline, ledger[key]

        t0 = time.time()
        pipeline, info = compute_fn()
        info = dict(info)
        info["stage"] = stage
        info["model"] = name
        info["wall_seconds"] = round(time.time() - t0, 2)

        with open(ckpt_file, "wb") as f:
            pickle.dump(pipeline, f)
        ledger[key] = info
        self.save(ledger)
        logger.info("[checkpoint] %s: computed and saved in %.2fs.", key, info["wall_seconds"])
        return pipeline, info


# ==============================================================================
# Baseline fitting (default hyperparameters, stage="baseline")
# ==============================================================================
def train_baselines(registry: dict, preprocessor: Pipeline, X_train, y_train, X_val, y_val,
                     ledger: CheckpointLedger) -> pd.DataFrame:
    """Fit every model in the registry with its library/config defaults
    (no tuning) and evaluate on the validation set. class_weight="balanced"
    is applied here because it's baked into the estimator at construction
    time in build_model_registry - it is a registry-level imbalance
    decision, not something this function decides on its own."""
    rows = []
    for name, spec in registry.items():

        def _compute_fn(name=name, spec=spec):
            pipe = Pipeline([("prep", clone(preprocessor)), ("model", clone(spec["estimator"]))])
            fit_kwargs = {}
            if spec["uses_sample_weight"]:
                fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)
            return fit_and_evaluate(pipe, X_train, y_train, X_val, y_val, fit_kwargs=fit_kwargs)

        _, info = ledger.checkpoint_or_run("baseline", name, _compute_fn)
        rows.append({"model": name, **info})

    baseline_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    baseline_df.insert(0, "rank", range(1, len(baseline_df) + 1))
    return baseline_df


# ==============================================================================
# Hyperparameter tuning (stage="tuned")
# ==============================================================================
def _tune_with_grid_search(name, spec, preprocessor, X_train, y_train, X_val, y_val, config):
    tune_cfg = config["tuning"]["grid_search"]
    pipe = Pipeline([("prep", clone(preprocessor)), ("model", clone(spec["estimator"]))])
    search = GridSearchCV(
        pipe, spec["param_grid"], cv=tune_cfg["cv"], scoring=tune_cfg["scoring"], n_jobs=tune_cfg["n_jobs"]
    )
    fit_kwargs = {}
    if spec["uses_sample_weight"]:
        fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)
    search.fit(X_train, y_train, **fit_kwargs)  # GridSearchCV refits the winner on all of X_train automatically

    metrics = evaluate_pipeline(search.best_estimator_, X_val, y_val)
    metrics["best_params"] = search.best_params_
    metrics["cv_pr_auc"] = float(search.best_score_)
    metrics["search_type"] = f"GridSearchCV(cv={tune_cfg['cv']}, full X_train)"
    return search.best_estimator_, metrics


def _tune_with_halving_subsample_refit(name, spec, preprocessor, X_train, y_train, X_val, y_val, config):
    """Compute-constrained strategy for expensive models (EDA Section 21):
    run HalvingRandomSearchCV on a stratified subsample of X_train to find a
    good configuration cheaply, then refit that WINNING configuration on the
    full training fold before it is ever scored or checkpointed."""
    tune_cfg = config["tuning"]["halving_random_search"]
    subsample_size = tune_cfg["subsample_size"]

    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=subsample_size, stratify=y_train,
        random_state=config["split"]["random_state"],
    )

    pipe = Pipeline([("prep", clone(preprocessor)), ("model", clone(spec["estimator"]))])
    search = HalvingRandomSearchCV(
        pipe, spec["param_distributions"],
        factor=tune_cfg["factor"], cv=tune_cfg["cv"], scoring=tune_cfg["scoring"],
        n_candidates=tune_cfg["n_candidates"], min_resources=tune_cfg["min_resources"],
        random_state=config["split"]["random_state"], n_jobs=tune_cfg["n_jobs"],
    )
    search_fit_kwargs = {}
    if spec["uses_sample_weight"]:
        search_fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train_sub)
    search.fit(X_train_sub, y_train_sub, **search_fit_kwargs)

    # Refit the WINNING configuration on the full training fold.
    refit_estimator = clone(spec["estimator"]).set_params(
        **{k.split("__", 1)[1]: v for k, v in search.best_params_.items()}
    )
    refit_pipe = Pipeline([("prep", clone(preprocessor)), ("model", refit_estimator)])
    refit_fit_kwargs = {}
    if spec["uses_sample_weight"]:
        refit_fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)
    fitted_pipe, metrics = fit_and_evaluate(refit_pipe, X_train, y_train, X_val, y_val, fit_kwargs=refit_fit_kwargs)

    metrics["best_params"] = search.best_params_
    metrics["subsample_cv_pr_auc"] = float(search.best_score_)
    metrics["search_type"] = (
        f"HalvingRandomSearchCV(subsample={subsample_size}) -> refit on full X_train"
    )
    return fitted_pipe, metrics


def tune_models(registry: dict, preprocessor: Pipeline, X_train, y_train, X_val, y_val,
                 config: dict, ledger: CheckpointLedger) -> pd.DataFrame:
    """Tune every tunable model with the strategy config.yaml assigns it.
    The naive benchmark (tunable=False) is skipped - it has no
    hyperparameters (EDA Section 20)."""
    rows = []
    for name, spec in registry.items():
        if not spec["tunable"]:
            continue

        method = spec["tuning_method"]
        if method == "grid_search":
            fn = _tune_with_grid_search
        elif method == "halving_random_search_subsample_refit":
            fn = _tune_with_halving_subsample_refit
        else:
            raise ValueError(f"Unknown tuning_method '{method}' for model '{name}'")

        def _compute_fn(name=name, spec=spec, fn=fn):
            return fn(name, spec, preprocessor, X_train, y_train, X_val, y_val, config)

        _, info = ledger.checkpoint_or_run("tuned", name, _compute_fn)
        rows.append({"model": name, **info})

    tuned_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    tuned_df.insert(0, "rank", range(1, len(tuned_df) + 1))
    return tuned_df


# ==============================================================================
# Final selection & one-time test evaluation
# ==============================================================================
def select_final_model(ledger: CheckpointLedger, registry: dict):
    """Scan EVERY checkpointed entry (both baseline and tuned stages, every
    model except the naive benchmark) and pick whichever has the highest
    validation PR AUC - does not assume the tuned version always wins."""
    all_ledger = ledger.load()
    naive_name = next(n for n, s in registry.items() if s["family"] == "naive")

    candidates = []
    for key, entry in all_ledger.items():
        stage, name = key.split("::", 1)
        if name == naive_name:
            continue
        candidates.append({"key": key, "stage": stage, "model": name, "pr_auc": entry["pr_auc"]})

    candidates_df = pd.DataFrame(candidates).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    best = candidates_df.iloc[0]

    ckpt_file = ledger.dir / f"{ledger._sanitize(best['key'])}.pkl"
    with open(ckpt_file, "rb") as f:
        final_model = pickle.load(f)

    return best, final_model, candidates_df


def evaluate_final_model(final_model: Pipeline, X_test, y_test) -> dict:
    """The ONE-TIME test-set evaluation. Call this exactly once, on the
    model selected by select_final_model, and never use X_test/y_test for
    anything else (tuning, comparison, model selection)."""
    return evaluate_pipeline(final_model, X_test, y_test)
