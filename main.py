"""Entry point: config -> data preparation -> patient-grouped split -> baselines -> tuning -> one-time test evaluation.

Every path (config, data, outputs, the src/ import) is resolved relative to THIS file, so `python main.py` behaves identically from any working directory.
  python main.py --self-test            data_preparation self-tests only
  python main.py --skip-tuning          baselines only (fast); aborts unless they reproduce the notebook baselines
  python main.py                        full run (tuning is resumable); aborts unless tuned parameters and test metrics reproduce the notebook
  python main.py --fresh                delete this run's ledger/checkpoints first
  python main.py --config PATH --output-dir PATH     (relative paths are resolved against this file's folder)
"""
import argparse, hashlib, json, os, platform, shutil, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd, scipy, sklearn  # noqa: E402
from threadpoolctl import threadpool_limits        # noqa: E402
import data_preparation as dp                      # noqa: E402
import model_training as mt                        # noqa: E402


def resolve(p): return p if os.path.isabs(p) else os.path.join(ROOT, p)


def close(a, b, rtol, atol): return abs(a - b) <= atol + rtol * abs(b)


def abort(msg): print("\nABORT - reproduction check failed:", msg); sys.exit(2)


def norm_params(d): return json.loads(json.dumps(d, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-tuning", action="store_true"); ap.add_argument("--fresh", action="store_true"); ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--config", default="src/config.yaml"); ap.add_argument("--output-dir", default=None)
    a = ap.parse_args(); t0 = time.time()
    cfg = dp.load_config(resolve(a.config))
    if a.self_test:
        dp.self_test(cfg, resolve(cfg["paths"]["raw_data"])); return
    out = resolve(a.output_dir or cfg["paths"]["output_dir"]); ref_dir = resolve(cfg["paths"]["notebook_reference_dir"]); data_path = resolve(cfg["paths"]["raw_data"]); O = cfg["outputs"]
    if os.path.abspath(out) == os.path.abspath(ref_dir): abort("production output dir must differ from the notebook's outputs folder")
    if a.fresh and os.path.exists(out): shutil.rmtree(out); print("--fresh: removed", out)
    os.makedirs(out, exist_ok=True); threadpool_limits(limits=cfg["tuning"]["blas_threads"]); mt.check_registry(cfg); tol = cfg["reproduction"]
    # ---------------------------------------------------------------- data preparation + split
    print("[1/5] data preparation"); raw = pd.read_csv(data_path); y_all = dp.normalise_target(raw[cfg["target"]["name"]], cfg["target"]["label_map"]); raw = dp.add_patient_key(raw, cfg)
    (i_tr, i_va, i_te), split = dp.make_group_split(raw, y_all, cfg); exp = cfg["expected_dims"]
    assert [len(i_tr), len(i_va), len(i_te)] == [exp["rows"][k] for k in ("train", "val", "test")], "split sizes differ from the notebook"
    feats = [c for c in raw.columns if c not in [cfg["target"]["name"]] + cfg["columns"]["raw_feature_exclude"]]; X_all, g_all = raw[feats], raw["patient"]
    X_tr, y_tr, g_tr, X_va, y_va, g_va = X_all.loc[i_tr], y_all.loc[i_tr], g_all.loc[i_tr], X_all.loc[i_va], y_all.loc[i_va], g_all.loc[i_va]
    print(f"  rows train/val/test = {len(i_tr)}/{len(i_va)}/{len(i_te)} | patient groups = {raw.loc[i_tr,'patient'].nunique()}/{raw.loc[i_va,'patient'].nunique()}/{raw.loc[i_te,'patient'].nunique()} (test locked)")
    # ---------------------------------------------------------------- baselines + reproduction check
    print("[2/5] baselines at library defaults"); fitted, base = mt.run_baselines(cfg, X_tr, y_tr, X_va, y_va, g_va); print(base.round(4).to_string(index=False))
    base.to_csv(os.path.join(out, O["baseline_results"]), index=False)
    nb_base = pd.read_csv(os.path.join(ref_dir, "baseline_results.csv")).set_index("model")
    for r in base.itertuples():
        for m in cfg["metrics"]:
            if not close(getattr(r, m), nb_base.loc[r.model, m], tol["rtol"], tol["atol"]): abort(f"baseline {r.model}.{m}: production {getattr(r, m)!r} vs notebook {nb_base.loc[r.model, m]!r}")
    print("  OK: baselines reproduce the notebook (relative+absolute tolerance rtol=%g atol=%g)" % (tol["rtol"], tol["atol"]))
    if a.skip_tuning: print(f"--skip-tuning: done in {time.time()-t0:.0f}s"); return
    # ---------------------------------------------------------------- tuning
    print("[3/5] tuning (resumable)"); ledger = mt.Ledger(out, cfg["paths"]["checkpoint_subdir"], cfg["paths"]["ledger_file"]); print("  ledger entries at start:", len(ledger.data))
    tuned = mt.run_tuning(cfg, X_tr, y_tr, g_tr, ledger, out)
    nb_params = json.load(open(os.path.join(ref_dir, "best_params.json")))
    for n, o in tuned.items():
        if norm_params(o["best_params"]) != norm_params(nb_params[n]): abort(f"tuned parameters for {n}: production {norm_params(o['best_params'])} vs notebook {norm_params(nb_params[n])}")
    print("  OK: tuned parameters reproduce the notebook")
    rows = []
    for n, o in tuned.items():
        v = mt.score(cfg, o["pipe"], X_va, y_va); b = base.set_index("model").loc[n]; rows.append(dict(model=n, base_pr_auc=b.pr_auc, tuned_pr_auc=v["pr_auc"], delta_pr_auc=v["pr_auc"] - b.pr_auc, cv_pr_auc=o["cv_mean"], cv_fold_sd=o["cv_sd"], secs=o["secs"]))
    pd.DataFrame(rows).to_csv(os.path.join(out, O["tuned_results"]), index=False); json.dump({k: v["best_params"] for k, v in tuned.items()}, open(os.path.join(out, O["best_params"]), "w"), indent=1, default=str)
    # ---------------------------------------------------------------- selection + one-time test
    print("[4/5] selection on validation, refit on train+validation, test evaluated once")
    cand, sel = mt.select_candidate(cfg, fitted, tuned, X_va, y_va); print(cand.round(5).to_string()); print("  SELECTED:", sel.model, sel.variant)
    TEST = mt.final_test(cfg, sel, tuned, X_tr, y_tr, X_va, y_va, X_all, y_all, g_all, i_te, ledger); pt, ci, paired = mt.evaluate_test(cfg, TEST)
    print(pd.DataFrame(pt).T.round(4).to_string())
    nb_final = json.load(open(os.path.join(ref_dir, "final_results.json")))
    if nb_final["selected"] != {"model": sel.model, "variant": sel.variant}: abort(f"selection differs: {sel.model}/{sel.variant} vs notebook {nb_final['selected']}")
    for k, v in nb_final["test"].items():
        for m, x in v.items():
            if not close(pt[k][m], x, tol["rtol"], tol["atol"]): abort(f"test metric {k}.{m}: production {pt[k][m]!r} vs notebook {x!r}")
    for m, (lo, hi) in nb_final["ci95_cluster_bootstrap"].items():
        if not (close(ci[m][0], lo, tol["rtol"], tol["atol"]) and close(ci[m][1], hi, tol["rtol"], tol["atol"])): abort(f"bootstrap CI for {m}: production {ci[m]} vs notebook {[lo, hi]}")
    print("  OK: selection, test metrics and bootstrap CIs reproduce the notebook")
    # ---------------------------------------------------------------- outputs
    print("[5/5] writing outputs to", out)
    ft = pd.DataFrame(pt).T; ft.index.name = "model"; ft.reset_index().to_csv(os.path.join(out, O["final_results"]), index=False)
    json.dump({"selected": {"model": sel.model, "variant": sel.variant}, "test": pt, "ci95_cluster_bootstrap": ci, "paired": paired}, open(os.path.join(out, O["final_results_json"]), "w"), indent=1)
    pd.DataFrame({"row": TEST["index"], "ID": TEST["ids"], "patient": TEST["groups"], "y_true": TEST["y"], "p_final": TEST["proba"]["final"], "pred_final": TEST["pred"]["final"]}).to_csv(os.path.join(out, O["test_predictions"]), index=False)
    pd.DataFrame({"row": raw.index, "ID": raw.ID, "patient": raw.patient, "split": split.values}).to_csv(os.path.join(out, O["split_ids"]), index=False)
    import pickle; pickle.dump(TEST["models"]["final"], open(os.path.join(out, O["final_model"]), "wb"))
    json.dump(dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, seed=cfg["split"]["seed"], compute=cfg["tuning"],
                   data_sha256=hashlib.sha256(open(data_path, "rb").read()).hexdigest(), ledger_keys=list(ledger.data), minutes=round((time.time() - t0) / 60, 2)), open(os.path.join(out, O["manifest"]), "w"), indent=1, default=str)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
