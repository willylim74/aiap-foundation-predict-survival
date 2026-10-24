# Predicting Coronary Artery Disease Patient Survival (`Survive`)

A patient-grouped, leakage-aware binary classification pipeline that predicts whether a coronary artery disease (CAD) patient survives, from `data/classification_bonus_practice_data.csv` (15,000 rows, 16 columns, 32.1% positive). The aim is to help doctors flag high-risk patients early.

**The one thing to know:** the 15,000 rows are about **230 patients replicated ~65 times each**. A random row split lets flexible models memorise patients (random CV gives HistGradientBoosting and RandomForest a ROC AUC of 1.000, versus 0.81-0.89 with patient-grouped CV). So everything here splits, cross-validates and bootstraps **by patient**. An earlier attempt (in `_archive/`) used a random split and reported a perfect test score; that result was leakage.

**Final model:** Logistic regression (C = 0.01) inside a leakage-safe sklearn Pipeline. On 35 unseen test patients it reaches **PR AUC 0.797** (95% CI [0.586, 0.941]), against 0.324 for the naive benchmark.

## Project layout

```
eda.ipynb                 Executed notebook: EDA (1A), cleaning/pipeline/baselines (1B), tuning + one-time test (1C)
main.py                   Entry point; every path resolves relative to this file
requirements.txt          Python dependencies
src/config.yaml           All parameters: paths, target, patient key, cleaning, features, split, models + grids, tuning, metrics, bootstrap
src/data_preparation.py   Cleaning, patient key, grouped split, transformers, preprocessing pipeline, self-tests
src/model_training.py     Baselines, tuning (with checkpoint ledger), selection, one-time test, bootstrap
data/                     Raw CSV
_archive/                 Superseded first attempt (random split, leaky)
```

`main.py` writes its results to `outputs_prod/`. It compares them with the notebook's reference results in `outputs/` (see the note under *Instructions for Executing the Pipeline*).

---

# Prerequisites and Installation

- **Python 3.12** (the delivered notebook ran on Python 3.12.10, pandas 3.0.5, numpy 2.5.2).
- Packages in `requirements.txt`: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, pyyaml, threadpoolctl (`threadpoolctl` is a scikit-learn dependency).
- Notebook only (not needed for `main.py`): jupyter, ipykernel (and nbformat/nbclient to re-execute headlessly).
- Note for pandas 3: text columns use the `str` dtype, so the pipeline selects categoricals with `dtype_exclude="number"` rather than `object`.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

# Instructions for Executing the Pipeline

Run from any directory (all paths resolve relative to `main.py`):

```bash
python main.py --self-test      # synthetic edge cases + real-data dimensions vs the notebook
python main.py --skip-tuning    # baselines only (~10 s)
python main.py                  # full run (~2.5 min, single CPU): tuning is resumable, test evaluated once
python main.py --fresh          # wipe this run's ledger/checkpoints first
python main.py --config PATH --output-dir PATH    # optional overrides (relative paths resolve against main.py)
```

- **Resumable:** tuning results are stored in a checkpoint ledger (`outputs_prod/ledger.json` + `checkpoints/`). An entry is reused only if its settings fingerprint matches. The final test evaluation is stored too, so re-runs never touch the test rows again.
- **Reproduction check:** `main.py` compares baselines, tuned parameters, the selected model, test metrics and bootstrap CIs with the notebook's results in `outputs/` (rtol = atol = 1e-6) and **aborts on mismatch**. `outputs/` is produced by running `eda.ipynb` and must exist for `--skip-tuning` and the full run. `--self-test` does not need it. `main.py` refuses to write into `outputs/`.
- **Changing parameters:** edit `src/config.yaml`, the single source of truth. Common edits: `split` (folds, seed), `cleaning` and `feature_engineering` (sentinels, cut-offs), `preprocessing` (winsor quantiles), `models` (classes and tuning grids), `tuning` (CV folds, halving budget), `selection` and `bootstrap`. Changing tuning settings changes the fingerprint, so the ledger recomputes. Changing the config will also trip the reproduction check, by design.
- **Outputs** (in `outputs_prod/`): `baseline_results.csv`, `tuned_results.csv`, `final_results.csv/.json`, `best_params.json`, `test_predictions.csv`, `split_ids.csv`, `final_model.pkl`, `run_manifest.json` (library versions, seed, data SHA-256, runtime) and `cv_results/`.

# Description of Logical Steps/Flow of the Pipeline

```
raw CSV
  │  normalise target (0/1/No/Yes -> 0/1; unknown labels raise)
  │  add patient key = |Age| | Sodium | Gender   (label-free)
  ▼
stratified, patient-grouped split 70/15/15  (StratifiedGroupKFold, 20 folds -> 14/3/3, seed 42)
  │   train 10,546 rows / 163 patients   validation 2,196 / 32   test 2,258 / 35 (locked)
  ▼
Preprocessing Pipeline (fitted on training rows only)
  DataCleaner -> FeatureEngineer -> ColumnTransformer
                                     ├─ numeric: Winsorizer -> median impute -> StandardScaler
                                     └─ categorical: OneHotEncoder(handle_unknown="ignore")
  ▼
Baselines at library defaults (Naive, Logistic, DecisionTree, RandomForest, HGB, KNN) -> validation
  ▼
Tuning on train only: patient-grouped 3-fold CV, scoring = PR AUC
  Logistic: grid | others: successive halving on a training subsample; winner refit on full train
  ▼
Selection on validation PR AUC (default vs tuned, rule fixed in advance; tie-break log-loss)
  ▼
Refit on train + validation -> evaluate on test ONCE
  ▼
95% patient-cluster bootstrap CIs, paired comparison vs default twin and naive benchmark
```

1. **Data preparation.** Fixes target spellings, derives the patient key and creates the grouped split. Test rows are not touched until the end.
2. **Cleaning and preprocessing.** All learned statistics (medians, winsor bounds, scaler, log columns, one-hot levels) live inside the Pipeline, so they are refit per model and per CV fold. No leakage.
3. **Baselines.** Genuine library defaults (seeds only, no class weights) set the reference, with train-validation gaps to expose memorisation.
4. **Tuning.** Regularisation-heavy search spaces, because unseen-patient accuracy falls with model complexity.
5. **Selection and test.** Pick by validation PR AUC, refit on train + validation, evaluate on test once.
6. **Uncertainty.** Patient-cluster bootstrap (1,000 resamples, seed 42) for every test metric and for paired differences.

# Overview of Key Findings from EDA

Details are in `eda.ipynb` (Phase 1A). Every target-related analysis used the training split only.

| Finding | Consequence |
|---|---|
| **~230 replicated patients** (32-197 rows each); no exact duplicates, yet 14,501 of 15,000 rows have a twin on six columns | Grouped split, grouped CV, patient-cluster bootstrap |
| **Leakage verdict: leaky.** Random CV gives RF/HGB ROC AUC 1.000; grouped CV gives 0.835 (HGB) and 0.888 (RF). Decision-tree accuracy rises to 0.999 under random CV but peaks at depth 2 (0.851) under grouped CV | Simple, regularised models preferred |
| Target stored in 4 spellings; `Smoke` case variants; `Ejection Fraction` `L`/`N`; 430 negative ages (sign flips) | Row-level cleaning, no rows removed |
| **Fill values**: Platelets = 263,358.03 (1,245 rows), CPK = 582 (2,331 rows); rows carrying them are more often positive, so the fill depends on the outcome | Converted to NaN; indicator flags kept **off** |
| **Creatinine missing (499 rows) only when outcome = 0** (not MCAR) | `Creatinine_missing` excluded; sensitivity checked, no gain on unseen patients |
| `ID` is non-unique (collisions, not repeats); `Favorite color` has no signal (eta² 0.0005) | Both dropped |
| Strongest signals: BMI (single-feature AUC 0.831), Weight, Creatinine, Age; sharp positive-rate jump above BMI ≈ 33 | BMI and threshold features engineered |
| Creatinine's high tail carries the signal (71.5% positive in extremes vs 29.7%) | Only gentle winsorising (0.5% / 99.5%) |
| Creatinine and CPK heavily right-skewed | log1p on columns whose train skew exceeds 1 |
| CPK and Platelets act as patient fingerprints, with little genuine signal | Not trusted for flexible models; grouped CV used |
| Only `Ejection Fraction` has a categorical association (eta² 0.010), and its cluster CI includes 0; striking Gender × Smoke cells rest on 4 patients | No categorical interactions engineered |
| `High` Ejection Fraction is **one patient**, entirely in validation | One-hot encoder uses `handle_unknown="ignore"` |
| Imbalance 2.12:1 (majority benchmark accuracy 0.679) | PR AUC as the ranking metric |
| No time/date column; row order and ID rank carry no information (AUC ≈ 0.50) | Time-ordered split not applicable |
| The data look simulated (spiky histograms, BMI-derived weights) | Treat scores as a property of this file |

# Feature Handling Description

| Feature | Handling | Used in model |
|---|---|---|
| `Survive` (target) | Spellings 0/1/No/Yes normalised to 0/1 | target |
| `ID` | Dropped (non-unique, no signal) | no |
| `Favorite color` | Dropped (no signal) | no |
| `Gender`, `Diabetes` | One-hot (unknown levels ignored) | yes |
| `Smoke` | Case-folded to `No`/`Yes`, then one-hot | yes |
| `Ejection Fraction` | `L`->`Low`, `N`->`Normal`, then one-hot | yes |
| `Age` | `abs()` to repair sign flips; winsorise; impute; scale | yes |
| `Sodium`, `Blood Pressure`, `Hemoglobin`, `Height`, `Weight` | Winsorise; median impute; standard scale | yes |
| `Creatinine` | Missing kept as NaN then median-imputed; log1p if train skew > 1; winsorise; scale | yes |
| `Platelets` | Fill value 263,358.03 -> NaN; median impute; log1p if train skew > 1; winsorise; scale | yes |
| `Creatine phosphokinase` | Fill value 582 -> NaN; median impute; log1p if train skew > 1; winsorise; scale | yes |
| `BMI` *(engineered)* | Weight / (Height/100)² | yes |
| `BMI_sq` *(engineered)* | BMI² | yes |
| `Age_x_BMI` *(engineered)* | Age × BMI | yes |
| `BMI_high` *(engineered)* | BMI ≥ 33 flag | yes |
| `Creatinine_high` *(engineered)* | Creatinine ≥ 1.5 flag | yes |
| `Creatinine_missing`, `Platelets_sentinel`, `CPK_sentinel` *(optional flags)* | Implemented but **off** (`cleaning.suspect_indicators: []`) because missingness depends on the outcome | no |
| `patient` (derived key) | Used only for splitting, CV and bootstrap, never as a feature | no |

The processed training matrix has **23 columns** (14 numeric + 9 one-hot) with no NaN. Which columns get log1p is learned in `fit` from training skew.

# Explanation of Model Choices

Five classifiers plus a naive benchmark are registered in `config.yaml`; a check fails if the registry differs from this list.

| Model | Why it is included | Tuning |
|---|---|---|
| **Naive** (majority class) | Benchmark; anything useful must beat PR AUC 0.324 | none |
| **Logistic regression** | Simple, regularised, interpretable; EDA showed capacity hurts on unseen patients | grid on `C` and `class_weight` (12 candidates) |
| **Decision tree** | Interpretable non-linear baseline; shows the memorisation curve | halving search |
| **Random forest** | Strong general-purpose ensemble; reduces variance | halving search |
| **HistGradientBoosting** | State-of-the-art tabular learner; handles NaN | halving search |
| **KNN** | Distance-based contrast; sensitive to replicated neighbours | halving search |

Tuning uses patient-grouped 3-fold CV scored by PR AUC. The grid is exhaustive for the linear model. Successive halving (300 -> 900 -> 2,700 -> 8,100 rows, 27 starting candidates) is used for the costlier models. Class weighting appears only in the tuning grids, never in the baselines.

**Selection rule (fixed before tuning):** the highest validation PR AUC across default and tuned variants, with validation log-loss as tie-break. The winner was **Logistic regression, C = 0.01** (validation PR AUC 0.711 vs 0.709 for its default twin). The preference for simple models matches the EDA: flexible models fit the training patients perfectly and generalise worse.

# Evaluation of Models

**Metrics.** PR AUC is the ranking metric: it suits the 2.1:1 imbalance and ignores true negatives. ROC AUC, F1, precision, recall and accuracy are also reported. Accuracy alone is weak because the majority benchmark already scores 0.679.

Split: 163 / 32 / 35 patients (train / validation / test), seed 42.

**Validation, library defaults**

| Model | PR AUC | ROC AUC | F1 | train-val PR gap |
|---|---|---|---|---|
| Logistic | 0.709 | 0.877 | 0.789 | 0.17 |
| RandomForest | 0.578 | 0.807 | 0.674 | 0.42 |
| HistGradientBoosting | 0.508 | 0.714 | 0.638 | 0.49 |
| DecisionTree | 0.500 | 0.728 | 0.634 | 0.50 |
| KNN | 0.472 | 0.689 | 0.536 | 0.53 |
| Naive (majority) | 0.324 | 0.500 | 0.000 | 0.00 |

**Tuning (validation PR AUC, default -> tuned):** Logistic 0.709 -> 0.711, DecisionTree 0.500 -> 0.558, RandomForest 0.578 -> 0.581, HGB 0.508 -> 0.661, KNN 0.472 -> 0.524. Only DecisionTree and HGB moved by more than one CV fold SD.

**Test (selected model refit on train + validation, evaluated once on 35 unseen patients)**

| Test | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Selected model (Logistic, C = 0.01) | 0.815 | 0.770 | 0.613 | 0.683 | 0.816 | 0.797 |
| 95% patient-cluster bootstrap CI | [0.701, 0.915] | [0.519, 1.000] | [0.353, 0.860] | [0.451, 0.854] | [0.663, 0.946] | [0.586, 0.941] |
| Default-parameter twin | 0.813 | 0.752 | 0.630 | 0.686 | 0.831 | 0.803 |
| Naive benchmark | 0.676 | 0.000 | 0.000 | 0.000 | 0.500 | 0.324 |

- The improvement over the naive benchmark is clear (PR AUC +0.473, CI [+0.302, +0.652]).
- Improvement over the default twin is about 0 with every CI including 0 (PR AUC -0.006 [-0.023, +0.005]): **tuning did not help the best model.**
- Calibration on test is reasonable on average (mean predicted 0.330 vs observed 0.324, Brier 0.140) but jagged by bin.
- Subgroups (descriptive, 2-27 patients each): strong for BMI ≥ 33; weak for BMI < 33 (ROC AUC 0.604, 25 patients); smokers are harder than non-smokers (ROC AUC 0.662 vs 0.888).

**Caveats**
- With only 35 test patients every metric has a wide CI, and rankings among neighbouring models are within noise.
- Two tuned optima sit on the edge of their grids (Logistic C = 0.01, HGB at its smallest capacity).
- The data look simulated, so 0.80 PR AUC is a property of this file, not clinical performance.
- Excluded `Creatinine_missing` and fill-value flags because the missingness mechanism depends on the outcome; if it is real in deployment, some signal is unused.
- A reduced-mode pre-flight notebook run (deleted) read the test split once before the delivered run; the selection rule was fixed beforehand and not changed.

# Considerations for Deployment

- **Not clinical-grade.** Training data are small (230 patients) and apparently simulated. Validate on real, independent patient cohorts before any clinical use; the model should support, not replace, clinician judgement.
- **Group-aware validation is mandatory.** Any retraining or monitoring must keep splitting by patient. A random row split will report inflated, meaningless scores.
- **Inputs and schema.** The saved Pipeline (`final_model.pkl`) takes the raw columns (including `ID` and `Favorite color`, which it drops) and performs cleaning, feature engineering, encoding and scaling itself. The `patient` key is not an input. Unseen categories (such as `Ejection Fraction = High`) are tolerated; unknown labels in the *target* raise during training.
- **Data quality at the source.** Fill values (Platelets 263,358.03, CPK 582), sign-flipped ages and casing variants are repaired by `DataCleaner`. Production should alert when new sentinel values appear, and fix them upstream where possible.
- **Missingness mechanism.** In this dataset Creatinine is missing only for non-survivors. If that mechanism does not exist in production, the model is unaffected (it does not use the flag), but missing values are median-imputed and may reduce accuracy.
- **Threshold and calibration.** Recall at the default 0.5 threshold is only 0.61. Pick the operating threshold from the costs of missed high-risk patients versus false alarms, and recalibrate on local data, since calibration is jagged across bins.
- **Fairness and subgroups.** Performance is weak for BMI < 33 and for smokers. Monitor subgroup performance and treat results for small groups with caution.
- **Performance and scalability.** A linear model on 23 features scores in microseconds and has negligible memory use. Real-time or batch scoring needs no special hardware.
- **Reproducibility and versioning.** `run_manifest.json` records library versions, seed, compute settings and the SHA-256 of the data. The pickled model depends on the scikit-learn version; pin versions or re-export for the serving environment, and never load pickles from untrusted sources.
- **Monitoring.** Track input drift (Age, Sodium, BMI distributions differed between train and validation), PR AUC on newly labelled patients with grouped bootstrap CIs, and retrain through `main.py` with the reproduction check updated to the new reference.
