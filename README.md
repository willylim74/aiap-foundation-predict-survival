# Predicting Coronary Artery Disease Patient Survival

The objective is to predict whether a coronary artery disease (CAD) patient survives (`Survive`), using the dataset provided, to help doctors identify high-risk patients early and formulate preemptive medical treatments.

We evaluate 8 baseline classification models (including a naive majority-class benchmark) and tune 7 of them via hyperparameter search.

#
# Prerequisites and Installation

Refer to **`requirements.txt`** for prerequisites needed to run the project. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`, `pyyaml`.

#
# Instructions for Executing the Pipeline
## main.py

Execute **`main.py`** directly from the repo root (expects `src/` alongside it, and the raw CSV at the path given by `data.raw_path` in `config.yaml`, default `data/data.csv`).

```bash
pip install -r requirements.txt
python main.py
```

In the **`config.yaml`** file, you can modify `models.registry` / `tuning` to change the search grid for each tuned model, and `output.base_dir` to control where results are written. Every fitted pipeline is checkpointed to `outputs/checkpoints/` as it's produced — re-running `main.py` resumes from there instead of recomputing anything already finished, so an interrupted run is safe to just run again.

#
# Description of Logical Steps/Flow of the Pipeline
## data_preparation.py

In **`data_preparation.py`**, we perform the logical steps and flow of our machine learning pipeline:
- data cleaning (row-level, run once, before the split)
- missing-value imputation (inside the preprocessing pipeline)
- outlier handling (inside the preprocessing pipeline)
- feature engineering (inside the preprocessing pipeline)
- preprocessing (scaling, encoding)

#
## Data Cleaning
- 15,000 patient records, 14 raw predictive features + `ID` (dropped — not a unique key, 885 collisions across the dataset, and carries no clinical value) + target `Survive`.
- No duplicate rows.
- `Survive` mixed two encodings on the same column — `0`/`1` alongside `"No"`/`"Yes"`/`"NO"`/`"YES"` — unified to a single 0/1 integer.
- `Smoke` mixed `No`/`Yes` with `NO`/`YES` casing — unified.
- `Ejection Fraction` mixed full labels (`Low`, `Normal`, `High`) with abbreviations (`L`, `N`) — unified to the full label.
- `Age` contained sign errors (430 negative values, ~2.9% of rows) — corrected via absolute value. The fix is justified rather than assumed: once made positive, those values land squarely inside the same range as the rest of the (genuine) ages, which is the signature of a sign-flip bug rather than a separate population.
- Missing values: `Creatinine` only (~3.3%), imputed inside the pipeline (median, fit on the training split only) with a `Creatinine_missing` indicator computed beforehand, in case missingness itself carries signal.

#
## Pre-Processing
The preprocessing pipeline consists of:

1. **Feature Engineering** (fit inside the pipeline, on the training split only)
   - `Gender_Smoke`, `Diabetes_Smoke`: explicit interaction features. `Gender` and `Smoke` each show ~no relationship to survival on their own, but combine into a strong, statistically robust interaction (female smokers 74.9% survival vs. 30.4% for female non-smokers; the direction reverses for men) — invisible to any model unless made explicit as a feature.
   - `Height` and `Favorite color` are **dropped** rather than engineered — both behave like synthetic/noise columns (`Height` is uniformly distributed with no bell shape at all; `Favorite color` is a near-perfectly even 6-way split), and neither shows any relationship to the target.

2. **Outlier Handling**
   - `IQRWinsorizer` (custom transformer): caps numeric values to `[Q1 - 3×IQR, Q3 + 3×IQR]`, fit on the training split only. A wider-than-standard multiplier (3× instead of the usual 1.5×) is used deliberately: extreme values in `Sodium`, `Creatinine` and `Platelets` are informative, not noise — rows with outlier values in these columns survive at dramatically different rates than typical rows (e.g. Sodium outliers: 77.8% survival vs. 31.5% typical) — so the pipeline caps their influence rather than removing them.

3. **Imputation**
   - `Creatinine`: median imputation, fit on the training split only.
   - All other numeric/categorical columns have no missing values.

4. **Feature Scaling** — numeric features: `StandardScaler`, applied after a `log1p` transform for the three heavily right-skewed columns (`Creatinine`, `Creatine phosphokinase`, `Platelets`; skew 1.5–4.4).

5. **Feature Encoding** — categorical features: One-hot (`Gender`, `Smoke`, `Diabetes`, `Ejection Fraction`, plus the engineered `Gender_Smoke` / `Diabetes_Smoke` interaction columns).

#
# Model Development
## model_training.py
#
## Data Splitting
- `Survive` is moderately imbalanced (~32.1% survive) — not severe, but enough to warrant class-weighting and a precision/recall-aware primary metric (see below).
- **Stratified 70/15/15 train/validation/test split.**
  - Training set: fitting each model.
  - Validation set: model comparison, hyperparameter tuning, and best-model selection.
  - Test set: final, once-only evaluation of the selected model.

#
## Train and Evaluate Baseline Models
Models: **Naive (majority class)**, **Logistic Regression**, **Ridge Classifier**, **Lasso (L1 Logistic)**, **Decision Tree**, **Random Forest**, **Gradient Boosting**, **K-Nearest Neighbors**. `class_weight="balanced"` is applied to every model that supports it directly; Gradient Boosting (no native `class_weight`) receives balanced `sample_weight` at fit time instead; KNN has no class-weighting mechanism and is used as-is.

Evaluation metrics: **Accuracy**, **Precision**, **Recall**, **F1**, **ROC-AUC**, **PR-AUC**.

Validation-set results from running this pipeline end-to-end on the full dataset:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.9996** | **1.000** | 0.9986 | **0.9993** | **1.0000** | **1.0000** |
| K-Nearest Neighbors | 0.9960 | 0.9945 | 0.9931 | 0.9938 | 0.9996 | 0.9984 |
| Decision Tree | 0.9956 | 0.9890 | 0.9972 | 0.9931 | 0.9960 | 0.9872 |
| Gradient Boosting | 0.9271 | 0.8570 | 0.9281 | 0.8911 | 0.9823 | 0.9674 |
| Ridge Classifier | 0.7791 | 0.6348 | 0.7358 | 0.6816 | 0.8590 | 0.7330 |
| Lasso (L1 Logistic) | 0.7804 | 0.6318 | 0.7593 | 0.6897 | 0.8600 | 0.7319 |
| Logistic Regression | 0.7804 | 0.6318 | 0.7593 | 0.6897 | 0.8599 | 0.7318 |
| Naive (majority class) | 0.6787 | 0.000 | 0.000 | 0.000 | 0.5000 | 0.3213 |

Random Forest is the strongest baseline on every metric, and already at a (near-)perfect PR-AUC — the flexible/non-linear models (Random Forest, KNN, Decision Tree) separate this dataset almost exactly, while the linear models plateau far lower (see *Key Findings from EDA*, point 7, for why).

#
## Hyperparameter Tuning
Hyperparameter search uses two different strategies depending on how expensive each model is to search, per `config.yaml`'s `models.registry` and `tuning` sections:

- **`GridSearchCV`** (exhaustive, `cv=5`, `scoring="average_precision"` i.e. PR-AUC, on the full training fold) for the models cheap enough to search directly: Logistic Regression, Ridge Classifier, Lasso, Decision Tree, K-Nearest Neighbors.
- **`HalvingRandomSearchCV`** with a compute-constrained **subsample-then-refit** strategy for the two expensive tree ensembles (Random Forest, Gradient Boosting): the search itself runs on a 1,500-row stratified subsample of the training fold (`factor=3`, `cv=3`, `n_candidates=10`), and the *winning* configuration is then refit on the **full** training fold before being scored or checkpointed — so the deployed model has seen as much data as every other model in the comparison, and the expensive part (many full-size tree fits) only happens once per model rather than once per candidate.

Tuned validation-set results:

| Model | PR-AUC (tuned) | PR-AUC (baseline) | Δ | Best Params |
|---|---|---|---|---|
| Gradient Boosting | **1.0000** | 0.9674 | +0.0326 | `n_estimators=150, max_depth=3, learning_rate=0.2, subsample=1.0` |
| Random Forest | 0.9996 | 1.0000 | −0.0004 | `n_estimators=150, max_depth=10, min_samples_leaf=1, max_features='sqrt'` |
| K-Nearest Neighbors | 0.9985 | 0.9984 | +0.0001 | `n_neighbors=5, weights='distance'` |
| Decision Tree | 0.9882 | 0.9872 | +0.0010 | `max_depth=None, min_samples_leaf=10` |
| Ridge Classifier | 0.7330 | 0.7330 | +0.0000 | `alpha=0.01` |
| Logistic Regression | 0.7320 | 0.7318 | +0.0002 | `C=10.0` |
| Lasso (L1 Logistic) | 0.7320 | 0.7319 | +0.0001 | `C=100.0` |

**Note on the result**: tuning helped Gradient Boosting meaningfully (closing most of its gap to the fully flexible models) and Decision Tree modestly, but left the linear models essentially unmoved and Random Forest/KNN unchanged since both were already saturated at baseline. Tuning is expected to yield only a modest improvement, not a dramatic one, on a dataset where the strongest baseline (Random Forest, PR-AUC 1.0000) is already at the ceiling.

#
## Final Model Evaluation
The best model is selected by **validation-set PR-AUC across all baseline and tuned models combined** (not tuned-only — this matters, see below), then evaluated once on the held-out **test set** for an unbiased final estimate.

**Selected model: Random Forest, baseline configuration** — validation PR-AUC 1.0000, narrowly ahead of tuned Gradient Boosting (0.999989). None of the tuning runs beat the plain-defaults Random Forest baseline; this is reported as the headline result rather than smoothed over (see *Key Findings from EDA*, point 7).

Final test-set evaluation (n=2,250, survive rate 32.1%):

| Metric | Value |
|---|---|
| Accuracy | 0.9987 |
| Precision | 1.0000 |
| Recall | 0.9958 |
| F1 | 0.9979 |
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |

Results are persisted to `outputs/`:
- `outputs/final_model.pkl` — the selected, fully-fitted pipeline (preprocessing + model), individually reloadable.
- `outputs/baseline_results.csv` / `outputs/tuned_vs_baseline_results.csv` — full-precision metrics for every model.
- `outputs/all_candidates_ranked.csv` — every baseline+tuned checkpoint, ranked by validation PR-AUC.
- `outputs/final_test_results.json` — test-set metrics for the selected model only.
- `outputs/final_summary_baseline_tuned_test.csv` — a single comparison table, with a flag column so the final selection is never ambiguous.
- `outputs/checkpoints/` — every fitted pipeline + a JSON ledger keyed `"<stage>::<model>"`, for resumability and audit.

#
# Key Findings from EDA
1. **Data overview**: 15,000 patient records. `Survive` is inconsistently encoded in the raw file (`0`/`1` mixed with `"No"`/`"Yes"`/`"NO"`/`"YES"`) and, once unified, is moderately imbalanced (~32.1% survive vs. ~67.9% do not, ≈2.1:1).
2. **Data quality issues found and fixed**: the same casing inconsistency affects `Smoke` and `Ejection Fraction`; `Age` has a sign-flip bug (430 negative values whose absolute value lands right back in the normal adult range); `ID` is not a unique key (885 collisions across 15,000 rows).
3. **Two columns behave like synthetic noise**: `Height` is uniformly distributed across its full range with essentially zero correlation to survival (r=0.006, p=0.45); `Favorite color` is a near-perfectly even 6-way split with no association to survival (ANOVA p=0.22). Both are dropped as features.
4. **A real interaction, invisible in either main effect**: `Gender` alone and `Smoke` alone both show ~no relationship to survival, but combined they show a strong, statistically robust interaction — female smokers survive far more than female non-smokers (74.9% vs. 30.4%), while for men the direction reverses. A milder version of the same pattern appears for `Diabetes × Smoke`.
5. **Extreme values are informative, not noise**: IQR-outlier rows in `Sodium`, `Creatinine` and `Platelets` survive at dramatically different rates than typical rows (e.g. Sodium outliers: 77.8% vs. 31.5%) — these are capped, never removed.
6. **`Ejection Fraction` is the one categorical feature with a real effect**: `Low` (33.9% survival) vs. `Normal` (19.2% survival) is a 77% relative difference, well-supported by sample size (13,216 vs. 1,724 rows).
7. **Model comparison reveals the data is close to a deterministic function of a few features**: Random Forest, KNN and Decision Tree all score near-perfectly (PR-AUC 0.99–1.00), while linear models plateau around PR-AUC 0.73. A depth-sweep diagnostic (fitting decision trees of increasing depth on just the raw numeric columns) shows validation accuracy climbing smoothly and consistently with model flexibility — 79% at depth 3 up to 99%+ unrestricted — with no widening train/validation gap, which rules out simple overfitting and instead points to `Survive` behaving like a near-deterministic rule over `Creatinine`, `Creatine phosphokinase`, `Platelets`, `Weight`, `Age` and `Sodium`. This is most likely a property of this being a partly synthetic practice dataset rather than evidence of real-world clinical predictability this clean — worth remembering when reading the near-1.0 PR-AUC scores above as a real-world performance claim.

#
# Explanation of Model Choices
This is a binary classification problem on a moderately imbalanced target. We selected a mix of linear and non-linear models, with the expectation that non-linear models would better capture the interaction and outlier structure the EDA surfaced (point 4, point 5 above) — which is exactly what happened.

Baseline models:
1. Logistic Regression — interpretable linear baseline.
2. Ridge Classifier — L2-penalized linear baseline.
3. Lasso (L1 Logistic) — L1-penalized linear baseline, the classification analogue of Lasso.
4. Decision Tree — single interpretable tree.
5. Random Forest — ensemble, generally robust to noisy/mixed-type features.
6. Gradient Boosting — sequential ensemble, typically strong on structured/tabular data.
7. K-Nearest Neighbors — simple non-parametric baseline.
8. Naive (majority class) — the floor every real model must beat.

Tuned via hyperparameter search: all of the above except the naive benchmark (which has no hyperparameters).

#
# Evaluation of Models
We evaluate models using:
1. Accuracy
2. Precision
3. Recall
4. F1
5. ROC-AUC
6. PR-AUC (**primary metric for model selection**)

PR-AUC is used as the primary selection metric rather than ROC-AUC because this is an imbalanced classification problem and the clinical objective specifically concerns identifying which patients will *not* survive — precision and recall are built around the minority class directly, while ROC-AUC's false-positive rate is computed against the much larger majority class too. PR-AUC also has a more interpretable baseline: a no-skill classifier scores PR-AUC equal to the positive-class prevalence (~0.321 here) rather than a constant 0.5, so a score of 1.0000 for the Random Forest baseline is directly readable as the maximum possible lift over "predict the majority class for everyone."

**Random Forest** is the strongest baseline and final performer across nearly every metric, most notably PR-AUC (1.0000) and ROC-AUC (1.0000) — a dramatic lift above the 0.321 no-skill baseline.

#
# Considerations for Deployment
- The near-perfect performance of the flexible models (Random Forest, KNN, Decision Tree — PR-AUC 0.99–1.00) is unusual enough for a real clinical prediction problem that it should not be taken at face value. EDA point 7 diagnoses this as `Survive` behaving like a near-deterministic function of a handful of raw lab values, consistent with this being a partly synthetic practice dataset. A model this confident on real hospital data would warrant real suspicion, not celebration, before any deployment decision — re-validate against genuine, independently-collected clinical data before treating these numbers as a real-world performance estimate.
- `Height` and `Favorite color` are dropped as features based on EDA evidence that they carry no signal on *this* dataset. If the pipeline is ever pointed at a different (even structurally similar) dataset, revisit `config.yaml`'s `preprocessing.noise_columns_to_drop` rather than assuming the same columns are uninformative there too.
- Hyperparameter tuning for the tree ensembles is compute-intensive even with the subsample-then-refit strategy — the full training-fold refit (10,500 rows, up to 250 estimators) is the dominant cost, at roughly 10–16 seconds per model in this environment. Budget accordingly, or reduce `param_distributions` / `subsample_size` in `config.yaml` for faster iteration during development.
- The final model selection process compares tuned **and untuned** checkpoints together, and in this run the untuned Random Forest baseline won outright. Any future re-run that only compares tuned models against each other (skipping the baseline stage) would silently pick a worse model — the two-stage, `stage::model_name`-namespaced checkpoint ledger in `outputs/checkpoints/` exists specifically to keep both comparisons available.
