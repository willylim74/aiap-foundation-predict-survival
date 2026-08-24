"""
main.py — Orchestration Layer

Controls the end-to-end flow of the CAD-survival classification pipeline:

    load config.yaml -> load data.csv -> data preparation -> model training
    -> final model evaluation

Run from the project root:

    python main.py

Every parameter this script touches (paths, target column, split ratios,
models, hyperparameter grids, metrics) comes from src/config.yaml — nothing
here is hardcoded — and every modeling decision traces back to a specific
finding in eda.ipynb (see the comments in src/config.yaml and
src/data_preparation.py / src/model_training.py for the EDA section each
one cites).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data_preparation import load_config, prepare_data  # noqa: E402
from src.model_training import (  # noqa: E402
    CheckpointLedger,
    build_model_registry,
    evaluate_final_model,
    select_final_model,
    split_data,
    train_baselines,
    tune_models,
)


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )


def main(config_path: str = "src/config.yaml") -> dict:
    # ------------------------------------------------------------------
    # 1. Load config.yaml
    # ------------------------------------------------------------------
    config = load_config(BASE_DIR / config_path)

    output_cfg = config["output"]
    _setup_logging(BASE_DIR / output_cfg["log_path"])
    logger = logging.getLogger("main")
    logger.info("=== CAD Survival Classification Pipeline ===")
    logger.info("Problem type: %s | Target: %s | Ranking metric: %s",
                config["project"]["problem_type"], config["data"]["target"],
                config["evaluation"]["ranking_metric"])

    # ------------------------------------------------------------------
    # 2. Load data + 3. Data preparation
    #    (prepare_data loads data/data.csv and applies the row-level
    #    structural cleaning from EDA Section 2/10; it also builds - but
    #    does not fit - the leakage-safe preprocessing pipeline from EDA
    #    Section 11/13.)
    # ------------------------------------------------------------------
    logger.info("--- Data preparation ---")
    df_clean, preprocessor = prepare_data(config, base_dir=BASE_DIR)
    logger.info("Cleaned data shape: %s", df_clean.shape)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean, config)

    # ------------------------------------------------------------------
    # 4. Model training
    #    (baseline fit -> hyperparameter tuning -> ranking, all checkpointed
    #    to a resumable ledger under output.checkpoints_dir)
    # ------------------------------------------------------------------
    logger.info("--- Model training: baselines ---")
    registry = build_model_registry(config)
    ledger = CheckpointLedger(BASE_DIR / output_cfg["checkpoints_dir"])

    baseline_df = train_baselines(registry, preprocessor, X_train, y_train, X_val, y_val, ledger)
    logger.info("Baseline results (ranked by %s):\n%s",
                config["evaluation"]["ranking_metric"], baseline_df.round(4).to_string(index=False))

    logger.info("--- Model training: hyperparameter tuning ---")
    tuned_df = tune_models(registry, preprocessor, X_train, y_train, X_val, y_val, config, ledger)
    logger.info("Tuned results (ranked by %s):\n%s",
                config["evaluation"]["ranking_metric"], tuned_df.round(4).to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Final model evaluation (one-time test-set evaluation)
    # ------------------------------------------------------------------
    logger.info("--- Final model selection & test evaluation ---")
    best, final_model, candidates_df = select_final_model(ledger, registry)
    logger.info("Selected model: %s (%s configuration), validation %s = %.4f",
                best["model"], best["stage"], config["evaluation"]["ranking_metric"], best["pr_auc"])

    test_metrics = evaluate_final_model(final_model, X_test, y_test)
    logger.info("Test-set evaluation for %s (%s): %s",
                best["model"], best["stage"], {k: round(v, 4) for k, v in test_metrics.items()})

    # ------------------------------------------------------------------
    # Persist every result table for reviewers
    # ------------------------------------------------------------------
    out_dir = BASE_DIR / output_cfg["base_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_df.to_csv(BASE_DIR / output_cfg["baseline_results_path"], index=False)
    tuned_df.to_csv(BASE_DIR / output_cfg["tuned_results_path"], index=False)
    candidates_df.to_csv(BASE_DIR / output_cfg["all_candidates_path"], index=False)

    final_summary_rows = []
    for name in registry:
        ledger_data = ledger.load()
        base_entry = ledger_data.get(f"baseline::{name}")
        tuned_entry = ledger_data.get(f"tuned::{name}")
        final_summary_rows.append({
            "model": name,
            "baseline_pr_auc": base_entry["pr_auc"] if base_entry else None,
            "tuned_pr_auc": tuned_entry["pr_auc"] if tuned_entry else None,
            "is_final_selection": name == best["model"],
        })
    import pandas as pd
    final_summary_df = pd.DataFrame(final_summary_rows)
    final_summary_df.to_csv(BASE_DIR / output_cfg["final_summary_path"], index=False)

    final_results = {
        "selected_model": best["model"],
        "selected_stage": best["stage"],
        "validation_pr_auc": float(best["pr_auc"]),
        "test_set_size": int(len(y_test)),
        "test_survive_rate": float(y_test.mean()),
        "test_metrics": test_metrics,
    }
    with open(BASE_DIR / output_cfg["final_test_results_path"], "w") as f:
        json.dump(final_results, f, indent=2)

    import pickle
    with open(BASE_DIR / output_cfg["final_model_path"], "wb") as f:
        pickle.dump(final_model, f)

    logger.info("All results written to %s", out_dir)
    logger.info("=== Pipeline complete ===")
    return final_results


if __name__ == "__main__":
    main()
