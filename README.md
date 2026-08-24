# CAD Survival Classification — End-to-End Pipeline

Predicts whether a coronary artery disease (CAD) patient survives
(`Survive`) from clinical, demographic and lifestyle attributes, so
clinicians can flag high-risk patients for preemptive treatment. This
project turns the exploratory analysis in [`eda.ipynb`](eda.ipynb) into a
runnable, config-driven pipeline — every cleaning step, feature, model, and
hyperparameter here traces back to a specific finding in that notebook.

## Project structure

```
root/
├── eda.ipynb                  Exploratory data analysis + baseline/tuned modeling (source of every decision below)
├── README.md                  This file
├── requirements.txt           Python dependencies
├── data/
│   └── data.csv                Raw dataset (15,000 patient records)
├── src/
│   ├── config.yaml             All configuration - paths, features, models, hyperparameters, metrics
│   ├── data_preparation.py     Row-level cleaning + leakage-safe preprocessing pipeline
│   └── model_training.py       Split, model registry, baseline fit, tuning, final test evaluation
└── main.py                     Orchestrates the pipeline end to end
```

## Running it

```bash
pip install -r requirements.txt
python main.py
```

`main.py` loads `src/config.yaml`, loads `data/data.csv`, runs data
preparation, trains and tunes every model in the registry, selects the best
one, evaluates it once on the held-out test set, and writes every result
table to `./outputs`. A second run resumes from `./outputs/checkpoints`
instead of recomputing anything already finished — safe to interrupt.

## What the EDA established (see `eda.ipynb` for full detail)

- **Problem type**: binary classification. `Survive` is inconsistently
  encoded in the raw file (`0`/`1` mixed with `"No"`/`"Yes"`/`"NO"`/`"YES"`)
  and, once unified, is imbalanced (~32% survive vs. ~68% do not, ≈2.1:1).
- **Data quality issues fixed before any modeling**: `Smoke` and
  `Ejection Fraction` have the same kind of casing/abbreviation
  inconsistency as `Survive`; `Age` has a sign-flip bug (430 negative
  values whose absolute value lands back in the normal adult range); `ID`
  is not a unique key (885 collisions across 15,000 rows) and is dropped as
  a feature; `Creatinine` is missing in 3.3% of rows (imputed, with a
  missingness indicator kept as its own feature).
- **Two columns are very likely synthetic noise** and are dropped:
  `Height` (uniformly distributed 130–189cm, no relationship to survival)
  and `Favorite color` (near-uniform 6-way split, no relationship to
  survival).
- **A real interaction, invisible in either main effect**: `Gender` and
  `Smoke` each look irrelevant to survival alone, but together show a
  strong, statistically robust interaction (female smokers survive far
  more than female non-smokers; the reverse for men). An engineered
  `Gender_Smoke` (and secondary `Diabetes_Smoke`) feature captures this.
- **Outliers in `Sodium`, `Creatinine` and `Platelets` are informative, not
  noise** (extreme rows have dramatically different survival rates) — so
  they are capped (winsorized), never removed.
- **Model comparison**: flexible models (Random Forest, KNN, Decision Tree)
  score near-perfectly on this dataset; linear models plateau far lower.
  Section 15.1 of the EDA diagnoses this as `Survive` behaving like a
  near-deterministic function of a handful of raw lab values — most likely
  a property of this being a partly synthetic practice dataset, not
  evidence of real-world clinical predictability this clean. The pipeline
  reproduces this honestly rather than hiding it.

## Configuration (`src/config.yaml`)

Every tunable decision in the pipeline lives here: data paths, the target
column, the feature list, the row-level cleaning rules, the preprocessing
pipeline's imputation/outlier/feature-engineering/encoding settings, the
70/15/15 stratified split, the full model registry (including the naive
majority-class benchmark), which hyperparameter search strategy each model
uses, cross-validation settings, the evaluation metrics and ranking metric
(PR AUC, appropriate for this imbalanced problem), and every output path.
Nothing in `data_preparation.py`, `model_training.py`, or `main.py`
hardcodes a column name, model, or hyperparameter — they all read this
file, so re-pointing the pipeline at a different (but structurally similar)
dataset, or changing a search grid, only requires editing this one file.

## Data preparation (`src/data_preparation.py`)

Two distinct stages, kept deliberately separate:

1. **Row-level structural cleaning** (`clean_raw_data`) — label unification,
   the `Age` sign-flip fix, a missingness indicator for `Creatinine`, and
   duplicate removal. Run once, on the full dataset, before any split,
   because none of it depends on a statistic estimated across rows.
2. **The preprocessing `Pipeline`** (`build_preprocessor`) — feature
   engineering (`InteractionFeatureAdder`), noise-column dropping
   (`NoiseColumnDropper`), and a `ColumnTransformer` that imputes, winsorizes
   (`IQRWinsorizer`) and log-transforms skewed numeric columns, imputes and
   scales the rest, passes through the missingness flag, and one-hot
   encodes categoricals (including the engineered interaction columns).
   This pipeline is only ever `fit()` on the training fold —
   `model_training.py` is responsible for that — which is what prevents
   validation/test data from leaking into any fitted statistic (imputation
   medians, winsorization bounds, scaling parameters).

## Model training (`src/model_training.py`)

- **Split**: stratified 70/15/15 train/validation/test (config-driven).
- **Model registry**: the naive majority-class benchmark plus Logistic
  Regression, Ridge Classifier, Lasso (L1 logistic), Decision Tree, Random
  Forest, Gradient Boosting and K-Nearest Neighbors — `class_weight="balanced"`
  applied to every model that supports it directly; Gradient Boosting (no
  native `class_weight`) gets balanced `sample_weight` at fit time instead.
- **Baselines**: every model fit with default/config hyperparameters,
  evaluated on the validation set.
- **Tuning**: `GridSearchCV` (exhaustive, full training fold) for the
  models cheap enough to search directly — Logistic, Ridge, Lasso, Decision
  Tree, KNN. A compute-constrained **subsample-then-refit** strategy with
  `HalvingRandomSearchCV` for the expensive tree ensembles — Random Forest,
  Gradient Boosting: search on a 1,500-row stratified subsample, then refit
  the winning configuration on the full training fold before it's scored.
- **Ranking metric**: PR AUC (average precision) — the appropriate metric
  for this imbalanced classification problem, not accuracy or plain ROC AUC.
- **Checkpointing**: every fitted pipeline is pickled and logged to
  `outputs/checkpoints/ledger.json`, keyed `"<stage>::<model>"` so the
  `"baseline"` and `"tuned"` stages can't collide on a shared model name.
  Re-running `main.py` skips anything already checkpointed.
- **Final selection**: scans *every* checkpointed entry across both stages
  and picks the single highest validation PR AUC — it does not assume
  tuning always wins (and, per the EDA, it often doesn't on this dataset).
  That model is evaluated **exactly once** on the untouched test set.

## Outputs

Running `main.py` populates `./outputs`:

| File | Contents |
|---|---|
| `baseline_results.csv` | Every model's default-hyperparameter metrics on validation |
| `tuned_vs_baseline_results.csv` | Every tunable model's tuned metrics + search details |
| `all_candidates_ranked.csv` | Every baseline+tuned checkpoint, ranked by validation PR AUC |
| `final_summary_baseline_tuned_test.csv` | Combined baseline/tuned comparison with the final selection flagged |
| `final_test_results.json` | The selected model's one-time test-set evaluation |
| `final_model.pkl` | The selected, fully-fitted pipeline (preprocessing + model) |
| `pipeline_run.log` | Full run log |
| `checkpoints/ledger.json` + `checkpoints/*.pkl` | Every fitted model + metadata, for resumability and audit |
