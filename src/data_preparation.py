"""Row-level cleaning, patient-grouped split, sklearn transformers and the preprocessing Pipeline.

Run `python data_preparation.py` (from any directory) to execute the embedded self-tests; they use synthetic edge cases and,
when the CSV is present, check the real-data dimensions against the notebook (numbers in config.yaml -> expected_dims).
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- row-level steps (no fitted statistics)
def load_config(path=None):
    import yaml
    with open(path or os.path.join(HERE, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalise_target(series, label_map):
    """'0'/'1'/'No'/'Yes' -> 0/1. Raises if any label is unknown (invalid targets are never silently dropped)."""
    y = series.astype(str).map(label_map)
    if y.isna().any():
        raise ValueError(f"unknown target labels: {sorted(series[y.isna()].astype(str).unique())}")
    return y.astype(int)


def add_patient_key(df, cfg):
    """Conservative patient key: |Age||Sodium|Gender. About 230 keys x ~65 rows in the real file."""
    pk = cfg["patient_key"]; parts = []
    for c in pk["columns"]:
        parts.append(df[c].abs().astype(str) if c in pk["abs_columns"] else df[c].astype(str))
    out = df.copy(); out["patient"] = parts[0].str.cat(parts[1:], sep="|"); return out


def make_group_split(df, y, cfg):
    """Stratified, patient-grouped 70/15/15: 20 StratifiedGroupKFold folds -> 14 train / 3 validation / 3 test folds."""
    s = cfg["split"]; fold = np.full(len(df), -1)
    for f, (_, te) in enumerate(StratifiedGroupKFold(s["n_group_folds"], shuffle=True, random_state=s["seed"]).split(df, y, df["patient"])):
        fold[te] = f
    assert (fold >= 0).all() and s["train_folds"] + s["val_folds"] + s["test_folds"] == s["n_group_folds"]
    perm = np.random.default_rng(s["seed"]).permutation(s["n_group_folds"])
    role = {f: ("train" if r < s["train_folds"] else "val" if r < s["train_folds"] + s["val_folds"] else "test") for r, f in enumerate(perm)}
    sp = pd.Series([role[f] for f in fold], index=df.index)
    idx = [sp.index[sp == k] for k in ("train", "val", "test")]
    g = [set(df.loc[i, "patient"]) for i in idx]
    assert not (set(idx[0]) & set(idx[1])) and not (set(idx[0]) & set(idx[2])) and not (set(idx[1]) & set(idx[2]))
    assert not (g[0] & g[1]) and not (g[0] & g[2]) and not (g[1] & g[2]), "a patient group appears in two splits"
    return idx, sp


# ----------------------------------------------------------------------------- transformers
class DataCleaner(BaseEstimator, TransformerMixin):
    """Stateless fixes, fill values -> NaN, optional outcome-suspect indicators, dropping columns. Learns nothing in fit."""
    def __init__(self, drop_cols=("ID", "Favorite color"), platelet_sentinel=263358.03, cpk_sentinel=582, ef_map=(("L", "Low"), ("N", "Normal")), indicators=()):
        self.drop_cols = drop_cols; self.platelet_sentinel = platelet_sentinel; self.cpk_sentinel = cpk_sentinel; self.ef_map = ef_map; self.indicators = indicators

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = np.asarray(self.transform(X.head(5)).columns, dtype=object); return self

    def transform(self, X):
        X = X.copy()
        X["Smoke"] = X["Smoke"].astype(str).str.capitalize()
        X["Ejection Fraction"] = X["Ejection Fraction"].astype(str).str.strip().replace(dict(self.ef_map))
        X["Age"] = X["Age"].abs()
        pm = np.isclose(X["Platelets"].astype(float), self.platelet_sentinel); cm = X["Creatine phosphokinase"] == self.cpk_sentinel
        ind = {"Creatinine_missing": X["Creatinine"].isna(), "Platelets_sentinel": pd.Series(pm, index=X.index), "CPK_sentinel": cm}
        for name in self.indicators: X[name] = ind[name].astype(int)
        X["Platelets"] = X["Platelets"].astype(float).mask(pm); X["Creatine phosphokinase"] = X["Creatine phosphokinase"].astype(float).mask(cm)
        return X.drop(columns=[c for c in self.drop_cols if c in X.columns])

    def get_feature_names_out(self, input_features=None): return self.feature_names_out_


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """BMI, BMI^2, Age*BMI, BMI>=bmi_cut, Creatinine>=creat_cut + log1p on columns whose TRAIN skew exceeds a threshold (learned in fit)."""
    def __init__(self, skew_threshold=1.0, candidates=("Creatinine", "Platelets", "Creatine phosphokinase"), bmi_cut=33.0, creat_cut=1.5):
        self.skew_threshold = skew_threshold; self.candidates = candidates; self.bmi_cut = bmi_cut; self.creat_cut = creat_cut

    def fit(self, X, y=None):
        self.log_cols_ = [c for c in self.candidates if c in X and X[c].min() >= 0 and X[c].skew() > self.skew_threshold]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = np.asarray(self.transform(X.head(5)).columns, dtype=object); return self

    def transform(self, X):
        X = X.copy(); bmi = X["Weight"] / (X["Height"] / 100) ** 2
        X["BMI"] = bmi; X["BMI_sq"] = bmi ** 2; X["Age_x_BMI"] = X["Age"] * bmi
        X["BMI_high"] = (bmi >= self.bmi_cut).astype(int); X["Creatinine_high"] = (X["Creatinine"] >= self.creat_cut).astype(int)
        for c in getattr(self, "log_cols_", []): X[c] = np.log1p(X[c])
        return X

    def get_feature_names_out(self, input_features=None): return self.feature_names_out_


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to the [lower, upper] quantiles of the FITTING data (NaN-safe)."""
    def __init__(self, lower=0.005, upper=0.995): self.lower = lower; self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X); self.cols_ = np.asarray(X.columns, dtype=object); self.lo_ = X.quantile(self.lower).values; self.hi_ = X.quantile(self.upper).values; return self

    def transform(self, X):
        X = pd.DataFrame(X); return X.clip(pd.Series(self.lo_, index=X.columns), pd.Series(self.hi_, index=X.columns), axis=1)

    def get_feature_names_out(self, input_features=None): return self.cols_


def build_preprocessing_pipeline(cfg, indicators=None):
    """UNFITTED Pipeline (fitted fresh per model / fold): cleaner -> engineer -> ColumnTransformer(numeric | one-hot, unknown ignored)."""
    cl, fe, pp = cfg["cleaning"], cfg["feature_engineering"], cfg["preprocessing"]
    ind = tuple(cl["suspect_indicators"] if indicators is None else indicators)
    num = Pipeline([("winsor", Winsorizer(pp["winsor_lower"], pp["winsor_upper"])), ("impute", SimpleImputer(strategy=pp["impute_strategy"])), ("scale", StandardScaler())])
    ct = ColumnTransformer([("num", num, selector(dtype_include="number")), ("cat", OneHotEncoder(handle_unknown="ignore"), selector(dtype_exclude="number"))])
    return Pipeline([("clean", DataCleaner(tuple(cfg["columns"]["drop"]), cl["platelet_sentinel"], cl["cpk_sentinel"], tuple(cl["ejection_fraction_map"].items()), ind)),
                     ("engineer", FeatureEngineer(fe["skew_threshold"], tuple(fe["log_candidates"]), fe["bmi_cut"], fe["creat_cut"])), ("encode", ct)])


# ----------------------------------------------------------------------------- self-tests
def _toy(cfg, n=240, seed=0):
    r = np.random.default_rng(seed); base = r.integers(0, 12, n)       # 12 replicated "patients"
    age = (40 + base * 4); sod = (130 + base)
    return pd.DataFrame({"ID": [f"A{i}" for i in range(n)], "Gender": np.where(base % 2 == 0, "Male", "Female"), "Smoke": r.choice(["No", "NO", "Yes", "YES"], n),
                         "Diabetes": r.choice(["Normal", "Diabetes"], n), "Age": age * r.choice([1, -1], n), "Ejection Fraction": r.choice(["Low", "L", "Normal", "N"], n), "Sodium": sod,
                         "Creatinine": np.where(r.random(n) < .1, np.nan, r.lognormal(0.2, 0.6, n)), "Platelets": np.where(r.random(n) < .1, cfg["cleaning"]["platelet_sentinel"], r.normal(260000, 60000, n)),
                         "Creatine phosphokinase": np.where(r.random(n) < .1, cfg["cleaning"]["cpk_sentinel"], r.lognormal(5.5, 1, n).astype(int)), "Blood Pressure": r.integers(40, 180, n),
                         "Hemoglobin": r.uniform(9, 17, n), "Height": r.integers(130, 190, n), "Weight": r.integers(30, 140, n), "Favorite color": r.choice(["red", "blue"], n)})


def self_test(cfg=None, data_path=None):
    from sklearn.base import clone
    cfg = cfg or load_config(); X = _toy(cfg)
    # --- cleaner
    out = DataCleaner(indicators=("Creatinine_missing", "Platelets_sentinel", "CPK_sentinel")).fit(X).transform(X)
    assert set(out["Smoke"]) <= {"No", "Yes"} and set(out["Ejection Fraction"]) <= {"Low", "Normal"} and (out["Age"] >= 0).all()
    assert "ID" not in out and "Favorite color" not in out
    assert out["Creatinine_missing"].sum() == X["Creatinine"].isna().sum()
    assert out["Platelets"].isna().sum() == out["Platelets_sentinel"].sum() > 0 and out["Creatine phosphokinase"].isna().sum() == out["CPK_sentinel"].sum() > 0
    assert "Creatinine_missing" not in DataCleaner().transform(X).columns, "suspect indicators must be OFF by default"
    # --- target + patient key
    assert normalise_target(pd.Series(["0", "1", "No", "Yes"]), cfg["target"]["label_map"]).tolist() == [0, 1, 0, 1]
    try: normalise_target(pd.Series(["maybe"]), cfg["target"]["label_map"]); raise SystemExit("unknown label accepted")
    except ValueError: pass
    k = add_patient_key(X, cfg); assert k["patient"].nunique() <= 12 and (add_patient_key(X.assign(Age=X.Age.abs()), cfg)["patient"] == k["patient"]).all(), "sign flips must not change the key"
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(X)) * 0 + (k["patient"].astype("category").cat.codes % 2).values)
    c3 = dict(cfg); c3["split"] = dict(cfg["split"], n_group_folds=6, train_folds=4, val_folds=1, test_folds=1)
    (a, b, c), sp = make_group_split(k, y, c3); assert len(a) + len(b) + len(c) == len(X) and len(set(k.loc[a, "patient"]) & set(k.loc[c, "patient"])) == 0
    # --- engineer / winsorizer
    fe = FeatureEngineer().fit(out); assert "Creatinine" in fe.log_cols_ and np.allclose(fe.transform(out)["BMI"], X["Weight"].values / (X["Height"].values / 100) ** 2)
    w = Winsorizer(0.1, 0.9).fit(pd.DataFrame({"a": np.arange(101.0)})); assert w.transform(pd.DataFrame({"a": [-5.0, 50, 500]})).iloc[:, 0].tolist() == [10.0, 50.0, 90.0]
    for T in (DataCleaner(), FeatureEngineer(), Winsorizer()): assert clone(T).get_params() == T.get_params()
    # --- pipeline: unseen / messy categories, no NaN, statistics from the fitting rows only
    Xf = X.drop(columns=[]); pp = build_preprocessing_pipeline(cfg).fit(Xf.iloc[:150])
    messy = Xf.iloc[150:160].copy(); messy.iloc[0, messy.columns.get_loc("Smoke")] = "YES "; messy.iloc[1, messy.columns.get_loc("Ejection Fraction")] = "High"; messy.iloc[2, messy.columns.get_loc("Diabetes")] = "Gestational"
    Z = pp.transform(messy); assert Z.shape[0] == 10 and not np.isnan(np.asarray(Z)).any()
    lo_a = pp.named_steps["encode"].named_transformers_["num"].named_steps["winsor"].lo_
    lo_b = build_preprocessing_pipeline(cfg).fit(Xf.iloc[60:]).named_steps["encode"].named_transformers_["num"].named_steps["winsor"].lo_
    assert not np.allclose(lo_a, lo_b), "fitted statistics must depend on the fitting rows"
    print("synthetic self-tests: PASS")
    # --- real-data dimensions vs the notebook (only when the CSV exists)
    path = data_path or os.path.join(os.path.dirname(HERE), cfg["paths"]["raw_data"])
    if os.path.exists(path):
        raw = pd.read_csv(path); yy = normalise_target(raw[cfg["target"]["name"]], cfg["target"]["label_map"]); kk = add_patient_key(raw, cfg)
        idx, _ = make_group_split(kk, yy, cfg); exp = cfg["expected_dims"]
        assert [len(i) for i in idx] == [exp["rows"][s] for s in ("train", "val", "test")], [len(i) for i in idx]
        assert [kk.loc[i, "patient"].nunique() for i in idx] == [exp["patient_groups"][s] for s in ("train", "val", "test")]
        feats = kk.drop(columns=[cfg["target"]["name"]] + cfg["columns"]["raw_feature_exclude"]); Zt = build_preprocessing_pipeline(cfg).fit(feats.loc[idx[0]]).transform(feats.loc[idx[0]])
        assert Zt.shape == (exp["rows"]["train"], exp["processed_columns"]) and not np.isnan(Zt).any(), Zt.shape
        print(f"real-data dimensions match the notebook: rows {[len(i) for i in idx]}, patient groups {[kk.loc[i,'patient'].nunique() for i in idx]}, matrix {Zt.shape}")
    else:
        print("real data not found; skipped dimension check")
    print("data_preparation self-tests: PASS")


if __name__ == "__main__":
    self_test()
