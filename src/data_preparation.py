"""
data_preparation.py — Data Preparation Layer

Cleans and preprocesses the raw CAD-survival dataset. Every decision here
traces back to a finding in eda.ipynb; see config.yaml for the specific
parameters (columns, thresholds, strategies) each step reads, and the
comments below for which EDA section justifies it.

Two distinct kinds of work happen here, deliberately kept separate:

1. Row-level structural cleaning (`clean_raw_data`) — deterministic,
   row-local fixes (label unification, a sign-flip bug fix, duplicate
   removal, a missingness indicator) that do not depend on any statistic
   estimated across rows. Safe to run ONCE on the full dataset, before the
   train/validation/test split.

2. The preprocessing Pipeline (`build_preprocessor`) — imputation, outlier
   capping, feature engineering, scaling and encoding. Every step in this
   pipeline is `fit()` exclusively on the training fold by model_training.py;
   nothing here ever estimates a statistic from validation/test data. This
   is what prevents data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ==============================================================================
# Config loading
# ==============================================================================
def load_config(config_path: str | Path) -> dict:
    """Load config.yaml into a plain dict. Shared by main.py, data_preparation.py
    and model_training.py so there is exactly one place that knows the file format."""
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


# ==============================================================================
# Data loading
# ==============================================================================
def load_raw_data(config: dict, base_dir: str | Path = ".") -> pd.DataFrame:
    """Load the raw CSV named in config['data']['raw_path'], relative to base_dir
    (base_dir is the project root when called from main.py)."""
    raw_path = Path(base_dir) / config["data"]["raw_path"]
    df = pd.read_csv(raw_path)
    logger.info("Loaded raw data: %s -> shape=%s", raw_path, df.shape)
    return df


# ==============================================================================
# 1. Row-level structural cleaning (run ONCE, before the split)
# ==============================================================================
def _unify_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Collapse inconsistent label encodings (EDA Section 2): Survive mixes
    "0"/"1" with "No"/"Yes"/"NO"/"YES"; Smoke mixes "No"/"Yes" with "NO"/"YES";
    Ejection Fraction mixes "Low"/"Normal"/"High" with abbreviations "L"/"N"."""
    df = df.copy()
    label_maps = config["data"]["label_maps"]

    survive_map = label_maps["Survive"]
    df["Survive"] = (
        df["Survive"].astype(str).str.strip().str.lower().map(survive_map)
    )
    if df["Survive"].isna().any():
        bad = df.loc[df["Survive"].isna(), "Survive"]
        raise ValueError(f"Unmapped Survive values found: {bad.unique().tolist()}")
    df["Survive"] = df["Survive"].astype(int)

    smoke_map = label_maps["Smoke"]
    df["Smoke"] = df["Smoke"].astype(str).str.strip().str.lower().map(smoke_map)

    ef_map = label_maps["Ejection Fraction"]
    df["Ejection Fraction"] = (
        df["Ejection Fraction"].astype(str).str.strip().str.lower().map(ef_map)
    )

    return df


def _fix_sign_flips(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """abs() the columns known to have a sign-flip data-entry bug (EDA Section 2:
    430 rows of Age were negative, and their absolute values land back in the
    normal adult range - the signature of a sign-flip bug, not a separate
    population or genuine outliers)."""
    df = df.copy()
    for col in config["cleaning"]["sign_flip_columns"]:
        df[col] = df[col].abs()
    return df


def _add_missingness_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add a binary <col>_missing flag for each column named in
    cleaning.missingness_indicator_columns (EDA Section 10). This is safe to
    compute before the split because it is purely row-local - it does NOT
    impute the underlying value (that happens inside the pipeline, fit on
    the training fold only)."""
    df = df.copy()
    for col in config["cleaning"]["missingness_indicator_columns"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def clean_raw_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run every row-level structural fix, once, on the full dataset -
    before any train/validation/test split. Order matters: labels are
    unified first so downstream logic never has to deal with mixed
    encodings, then structural bugs are fixed, then missingness is flagged,
    then duplicates and the identifier column are handled."""
    n_before = len(df)

    df = _unify_labels(df, config)
    df = _fix_sign_flips(df, config)
    df = _add_missingness_indicators(df, config)

    if config["cleaning"]["drop_duplicates"]:
        n_dupes = df.duplicated().sum()
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Dropped %d fully-duplicated rows (EDA found 0 on this dataset - "
                     "kept for robustness against future data refreshes)", n_dupes)

    id_col = config["data"]["id_column"]
    if id_col in df.columns:
        # Kept as a column (never dropped from the frame) so callers can
        # still look up a row by ID, but it is excluded from the feature
        # matrix in model_training.split_data - EDA Section 2/10 found it is
        # not a unique key (885 collisions) and carries no clinical signal.
        pass

    logger.info("clean_raw_data: %d -> %d rows after cleaning", n_before, len(df))
    return df


# ==============================================================================
# 2. Feature-engineering transformers (scikit-learn-compatible)
#    Mirrors the classes validated in eda.ipynb Phase 1B Section 11.
# ==============================================================================
class InteractionFeatureAdder(BaseEstimator, TransformerMixin):
    """Adds combined categorical interaction columns (e.g. Gender_Smoke)
    from config['preprocessing']['interaction_features']. EDA Section 7:
    Gender and Smoke each look irrelevant to survival alone but combine into
    a strong interaction (female smokers 74.9% survival vs. 30.4% for female
    non-smokers; reversed direction for men) that neither parent column can
    express on its own."""

    def __init__(self, interactions: list[dict] | None = None):
        self.interactions = interactions or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for spec in self.interactions:
            name = spec["name"]
            comps = spec["components"]
            X[name] = X[comps[0]].astype(str)
            for c in comps[1:]:
                X[name] = X[name] + "_" + X[c].astype(str)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + [spec["name"] for spec in self.interactions])


class NoiseColumnDropper(BaseEstimator, TransformerMixin):
    """Drops columns identified in EDA as non-informative / likely-synthetic
    (default from config: Height, Favorite color - EDA Sections 3-5)."""

    def __init__(self, cols: list[str] | None = None):
        self.cols = cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns])


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """Clips numeric columns to [Q1 - k*IQR, Q3 + k*IQR], with bounds fit
    only on the data passed to .fit() - i.e. the training fold only, once
    this is used inside a Pipeline. EDA Section 8: IQR-outlier rows in
    Sodium/Creatinine/Platelets survive at dramatically different rates than
    typical rows, so the right handling is capping their influence, not
    deleting the rows - k=3.0 (wider than the standard 1.5) by default so
    only genuinely extreme values are touched."""

    def __init__(self, multiplier: float = 3.0):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lo_ = q1 - self.multiplier * iqr
        self.hi_ = q3 + self.multiplier * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lo_, self.hi_)


# ==============================================================================
# 3. The preprocessing Pipeline
# ==============================================================================
def build_preprocessor(config: dict) -> Pipeline:
    """Build (but do not fit) the full leakage-safe preprocessing pipeline:
    feature engineering -> noise-column drop -> per-branch impute/winsorize/
    (log-transform)/scale for numeric columns, impute/one-hot for
    categoricals, passthrough for the missingness flag. The caller (
    model_training.py) is responsible for calling .fit() only on the
    training fold."""
    prep_cfg = config["preprocessing"]

    skewed_num = prep_cfg["numeric_columns"]["skewed"]
    normal_num = prep_cfg["numeric_columns"]["normal"]
    passthrough_cols = prep_cfg["passthrough_columns"]
    cat_cols = prep_cfg["categorical_columns"]

    num_strategy = prep_cfg["missing_values"]["numeric_strategy"]
    cat_strategy = prep_cfg["missing_values"]["categorical_strategy"]
    winsor_multiplier = prep_cfg["outliers"]["multiplier"]

    column_transformer = ColumnTransformer(
        transformers=[
            ("skewed_num", Pipeline([
                ("impute", SimpleImputer(strategy=num_strategy)),
                ("winsorize", IQRWinsorizer(multiplier=winsor_multiplier)),
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]), skewed_num),
            ("normal_num", Pipeline([
                ("impute", SimpleImputer(strategy=num_strategy)),
                ("winsorize", IQRWinsorizer(multiplier=winsor_multiplier)),
                ("scale", StandardScaler()),
            ]), normal_num),
            ("flag", "passthrough", passthrough_cols),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy=cat_strategy)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        sparse_threshold=0,
    )

    preprocessor = Pipeline([
        ("feature_engineering", InteractionFeatureAdder(prep_cfg["interaction_features"])),
        ("drop_noise", NoiseColumnDropper(prep_cfg["noise_columns_to_drop"])),
        ("column_transform", column_transformer),
    ])
    return preprocessor


# ==============================================================================
# Convenience entry point used by main.py
# ==============================================================================
def prepare_data(config: dict, base_dir: str | Path = ".") -> tuple[pd.DataFrame, Pipeline]:
    """Load the raw CSV, apply row-level structural cleaning, and build (but
    do not fit) the preprocessing pipeline. Returns the cleaned dataframe
    (target + features + ID, ready to be split) and the unfitted
    preprocessing Pipeline."""
    df_raw = load_raw_data(config, base_dir=base_dir)
    df_clean = clean_raw_data(df_raw, config)
    preprocessor = build_preprocessor(config)
    return df_clean, preprocessor
