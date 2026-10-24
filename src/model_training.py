"""Baselines, tuning with a checkpoint ledger, selection and the one-time test evaluation. Everything is driven by config.yaml."""
import hashlib, json, os, pickle, time
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from data_preparation import build_preprocessing_pipeline

CLASSES = {c.__name__: c for c in (DummyClassifier, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, HistGradientBoostingClassifier, KNeighborsClassifier)}
REQUIRED = ["Logistic", "DecisionTree", "RandomForest", "HistGradientBoosting", "KNN"]


def check_registry(cfg):
    assert set(cfg["models"]) == set(REQUIRED) | {"Naive"}, "config registry must equal the required model list + the naive benchmark"


def make_pipe(cfg, name, indicators=None):
    m = cfg["models"][name]
    return Pipeline([("prep", build_preprocessing_pipeline(cfg, indicators)), ("model", CLASSES[m["class"]](**m["params"]))])


def score(cfg, pipe, X, y):
    p = pipe.predict(X); s = pipe.predict_proba(X)[:, 1]
    return dict(accuracy=accuracy_score(y, p), precision=precision_score(y, p, zero_division=0), recall=recall_score(y, p), f1=f1_score(y, p),
                roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s))


def cluster_boot(y, s, groups, B=300, seed=42):
    y, s = np.asarray(y), np.asarray(s); u, inv = np.unique(np.asarray(groups), return_inverse=True); idx = [np.where(inv == i)[0] for i in range(len(u))]
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        ix = np.concatenate([idx[i] for i in rng.integers(0, len(u), len(u))])
        if y[ix].min() != y[ix].max(): out.append(average_precision_score(y[ix], s[ix]))
    return np.percentile(out, [2.5, 97.5])


def run_baselines(cfg, X_tr, y_tr, X_va, y_va, g_va, log=print):
    """Genuine library defaults (config params only: seeds). Class weighting is a tuning-only option."""
    fitted, rows = {}, []
    for name in cfg["models"]:
        t = time.time(); p = make_pipe(cfg, name).fit(X_tr, y_tr); fitted[name] = p
        v, tr = score(cfg, p, X_va, y_va), score(cfg, p, X_tr, y_tr); ci = cluster_boot(y_va, p.predict_proba(X_va)[:, 1], g_va, seed=cfg["split"]["seed"])
        rows.append(dict(model=name, **{m: v[m] for m in cfg["metrics"]}, train_pr_auc=tr["pr_auc"], gap_pr_auc=tr["pr_auc"] - v["pr_auc"], pr_auc_ci_low=ci[0], pr_auc_ci_high=ci[1], fit_seconds=time.time() - t))
        log(f"  baseline {name:22s} val PR AUC {v['pr_auc']:.4f}")
    base = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True); base.insert(0, "rank", base.index + 1)
    return fitted, base


# ----------------------------------------------------------------------------- checkpoint ledger
def fingerprint(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Ledger:
    """JSON ledger + pickle checkpoints, keys 'stage::model'. An entry is reused only if its settings fingerprint matches."""
    def __init__(self, out_dir, ckpt_subdir, ledger_file):
        self.ckpt = os.path.join(out_dir, ckpt_subdir); self.path = os.path.join(out_dir, ledger_file); os.makedirs(self.ckpt, exist_ok=True)
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def _p(self, key): return os.path.join(self.ckpt, key.replace("::", "__") + ".pkl")

    def get(self, key, fp):
        if key in self.data and self.data[key].get("fingerprint") == fp and os.path.exists(self._p(key)): return pickle.load(open(self._p(key), "rb"))
        return None

    def put(self, key, fp, obj, report):
        pickle.dump(obj, open(self._p(key), "wb")); self.data[key] = {"fingerprint": fp, "report": report, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(self.data, open(self.path, "w"), indent=1, default=str)


def _cv(cfg): return StratifiedGroupKFold(cfg["tuning"]["cv_folds"], shuffle=True, random_state=cfg["split"]["seed"])


def tuning_fingerprint(cfg, name, n_train):
    t = cfg["tuning"]; return fingerprint(dict(space=cfg["models"][name]["grid"], search=cfg["models"][name]["search"], n_train=n_train, seed=cfg["split"]["seed"], folds=t["cv_folds"], halving=t["halving"], pipeline=t["pipeline_version"]))


def run_tuning(cfg, X_tr, y_tr, g_tr, ledger, out_dir, log=print):
    """Grid search for the linear model, successive halving (resource = n_samples) on a training subsample for the others; the winner is refit on the FULL training split.
    Tuned settings are applied only here (set_params on a clone); baselines are never touched."""
    t = cfg["tuning"]; h = t["halving"]; seed = cfg["split"]["seed"]; sub_n = h["min_resources"] * h["factor"] ** 3; tuned = {}
    os.makedirs(os.path.join(out_dir, "cv_results"), exist_ok=True)
    for name, m in cfg["models"].items():
        if "search" not in m: continue
        key, fp = f"tune::{name}", tuning_fingerprint(cfg, name, len(X_tr)); hit = ledger.get(key, fp)
        if hit is not None: log(f"  [resume] {key}"); tuned[name] = hit; continue
        t0 = time.time(); pipe = make_pipe(cfg, name)
        if m["search"] == "grid":
            s = GridSearchCV(pipe, m["grid"], scoring=t["scoring"], cv=_cv(cfg), n_jobs=t["n_jobs"], refit=False); s.fit(X_tr, y_tr, groups=g_tr)
        else:
            sub = X_tr.sample(min(sub_n, len(X_tr)), random_state=seed)
            s = HalvingRandomSearchCV(pipe, m["grid"], resource=h["resource"], min_resources=h["min_resources"], factor=h["factor"], n_candidates=h["n_candidates"], scoring=t["scoring"],
                                      cv=_cv(cfg), refit=False, random_state=seed, n_jobs=t["n_jobs"])
            s.fit(sub, y_tr.loc[sub.index], groups=g_tr.loc[sub.index])
        cvr = pd.DataFrame(s.cv_results_); bi = s.best_index_; winner = clone(pipe).set_params(**s.best_params_).fit(X_tr, y_tr)
        out = dict(pipe=winner, best_params=s.best_params_, cv_mean=float(cvr.loc[bi, "mean_test_score"]), cv_sd=float(cvr.loc[bi, "std_test_score"]), cv_results=cvr, secs=time.time() - t0)
        report = f"{name}: best={s.best_params_} | grouped-CV PR AUC {out['cv_mean']:.4f} +- {out['cv_sd']:.4f} | {out['secs']:.1f}s"
        ledger.put(key, fp, out, report); cvr.drop(columns=["params"], errors="ignore").to_csv(os.path.join(out_dir, "cv_results", f"{name}.csv"), index=False); log("  [done] " + report); tuned[name] = out
    return tuned


# ----------------------------------------------------------------------------- selection + one-time test
def select_candidate(cfg, fitted, tuned, X_va, y_va):
    rows = []
    for variant, pool in (("default", {n: p for n, p in fitted.items() if n != "Naive"}), ("tuned", {n: o["pipe"] for n, o in tuned.items()})):
        for n, p in pool.items():
            sv = p.predict_proba(X_va)[:, 1]; rows.append((n, variant, average_precision_score(y_va, sv), log_loss(y_va, np.clip(sv, 1e-6, 1 - 1e-6)), roc_auc_score(y_va, sv)))
    C = pd.DataFrame(rows, columns=["model", "variant", "val_pr_auc", "val_logloss", "val_roc_auc"]).sort_values(["val_pr_auc", "val_logloss"], ascending=[False, True]).reset_index(drop=True)
    return C, C.iloc[0]


def final_test(cfg, sel, tuned, X_tr, y_tr, X_va, y_va, X_all, y_all, g_all, i_te, ledger, log=print):
    """Refit on train+validation; the TEST rows are read here and only here; the result is stored in the ledger so a re-run reuses it."""
    X_tv, y_tv = pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va])
    params = tuned[sel.model]["best_params"] if sel.variant == "tuned" else None
    fp = fingerprint(dict(sel=(sel.model, sel.variant), params=params, n=len(X_tv), seed=cfg["split"]["seed"]))
    TEST = ledger.get("final::test", fp)
    if TEST is not None: log("  [resume] final::test already evaluated: stored predictions reused, test rows not touched"); return TEST
    def sel_pipe():
        p = make_pipe(cfg, sel.model)
        return p.set_params(**params) if params else p
    X_te, y_te, g_te = X_all.loc[i_te], y_all.loc[i_te], g_all.loc[i_te]
    models = {"final": sel_pipe().fit(X_tv, y_tv), "twin": make_pipe(cfg, sel.model).fit(X_tv, y_tv), "naive": make_pipe(cfg, "Naive").fit(X_tv, y_tv)}
    TEST = dict(index=np.asarray(X_te.index), ids=X_te["ID"].values, groups=g_te.values, y=y_te.values, pred={k: m.predict(X_te) for k, m in models.items()},
                proba={k: m.predict_proba(X_te)[:, 1] for k, m in models.items()}, models=models, selected=(sel.model, sel.variant))
    ledger.put("final::test", fp, TEST, f"final test evaluated once: {sel.model} ({sel.variant}) on {len(y_te)} rows"); log("  final test evaluated (once)"); return TEST


def evaluate_test(cfg, TEST):
    y, g = TEST["y"], TEST["groups"]; names = cfg["metrics"]
    def met(yy, pr, s): return [accuracy_score(yy, pr), precision_score(yy, pr, zero_division=0), recall_score(yy, pr), f1_score(yy, pr), roc_auc_score(yy, s), average_precision_score(yy, s)]
    pt = {k: dict(zip(names, met(y, TEST["pred"][k], TEST["proba"][k]))) for k in ("final", "twin", "naive")}
    u, inv = np.unique(g, return_inverse=True); members = [np.where(inv == i)[0] for i in range(len(u))]; rng = np.random.default_rng(cfg["bootstrap"]["seed"]); boot = {k: [] for k in pt}
    for _ in range(cfg["bootstrap"]["n"]):
        ix = np.concatenate([members[i] for i in rng.integers(0, len(u), len(u))])
        if y[ix].min() == y[ix].max(): continue
        for k in pt: boot[k].append(met(y[ix], TEST["pred"][k][ix], TEST["proba"][k][ix]))
    boot = {k: np.array(v) for k, v in boot.items()}
    ci = {m: [float(x) for x in np.percentile(boot["final"][:, j], [2.5, 97.5])] for j, m in enumerate(names)}
    paired = {f"{o}|{m}": [float(pt["final"][m] - pt[o][m])] + [float(x) for x in np.percentile(boot["final"][:, j] - boot[o][:, j], [2.5, 97.5])] for o in ("twin", "naive") for j, m in enumerate(names)}
    return pt, ci, paired
