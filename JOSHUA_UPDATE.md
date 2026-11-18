# PathSnap — Joshua's models, training and tuning update

This update is based on the repository's current `main` branch. I re-checked that `src/features.py` and `src/evaluate.py` are now present.

## Compatibility with the latest team code

- `src/features.py` exposes `FEATURE_COLUMNS`, `load_or_build(partition, rebuild=False)`, and `save_feature_list(path)`.
- `src/evaluate.py` exposes `group_cv(factory, params, cand, fixes, features, n_folds, max_train_rows, seed, verbose)`, `training_rows(...)`, and `evaluate_predictions(...)`.
- `src/config.py` exposes `SEED`, `N_FOLDS`, `PROCESSED_DIR`, `ARTIFACTS_DIR`, and `RESULTS_DIR`.

This update does not replace or alter your teammate's feature-engineering/evaluation modules. It integrates directly with their interfaces. In particular, `src/model_zoo.py` factories accept keyword parameters because the shared evaluator calls `factory(**params)`. The tuner now calls the shared `src.evaluate.group_cv`; it no longer maintains a separate CV implementation.

## Files to copy into the repository

- `src/model_zoo.py` — Random Forest and XGBoost model factories and registry.
- `src/train.py` — training/evaluation CLI that calls the team's shared GroupKFold evaluator.
- `src/tune.py` — Optuna TPE Bayesian optimisation for Random Forest/XGBoost, using the same shared evaluator.
- `src/model_selection.py` — ranks completed model comparisons; won't label a winner final until all six team classifiers are present.
- `tests/test_model_workflow.py` — regression tests for the model/evaluation/tuning integration.
- `requirements.txt` — the current main-branch requirements with `optuna==4.8.0` added.

## Commands

Run these from the repository root:

```bash
pip install -r requirements.txt

# 1. Generate candidates and fix tables
python -m src.build_dataset

# 2. Generate feature parquet files (train + student_test)
python -m src.features --partition all

# 3. Train the two models on real training trips and evaluate on student_test
python -m src.train --model random_forest
python -m src.train --model xgboost

# 4. Rank the available comparison rows (provisional until all team models are present)
python -m src.model_selection

# 5. Tune the model after its baseline is trained/compared
python -m src.tune --model xgboost --n-trials 40
# or
python -m src.tune --model random_forest --n-trials 40
```

`src.train` defaults to all valid training candidate rows. To run a smaller diagnostic fit, pass e.g. `--max-train-rows 250000`; record the cap when comparing results, because a capped fit is not directly equivalent to a full-data fit.

Optuna tuning defaults to 40 trials and uses a maximum of 300,000 candidate rows sampled from each CV training fold. The full PathSnap training data has roughly 1.87 million candidate rows, so 40 trials × 5 folds on every row can be very expensive. Validation metrics are still evaluated on all rows in each held-out trip. The final tuned refit and student-test evaluation use all valid training candidate rows. To use the full dataset inside every tuning fold, pass `--full-fold-data`.

For an initial wiring check, run only `--n-trials 2`, then use 30–50 trials for the proper experiment.

## Outputs

Training outputs:

- `results/models/<model>/config.json`
- `results/models/<model>/cv_folds.csv`
- `results/models/<model>/student_test_metrics.json`
- `results/models/<model>/model.joblib`
- `results/models/model_comparison.csv`
- `artifacts/features.json`

Model selection outputs:

- `results/models/model_selection_ranking.csv`
- `results/models/model_selection.json`

Tuning outputs:

- `results/tuning/<model>/optuna_trials.csv`
- `results/tuning/<model>/optimization_history.png`
- `results/tuning/<model>/parameter_importance.png`
- `results/tuning/<model>/untuned_cv_folds.csv` and `tuned_cv_folds.csv`
- `results/tuning/<model>/tuned_vs_untuned.csv`
- `results/tuning/<model>/best_params.json`
- `results/tuning/<model>/tuned_model.joblib`
- resumable SQLite Optuna database

The tuned estimator remains under `results/tuning/<model>/` for now. The later export task should select the approved model and write `artifacts/model.joblib`, `artifacts/features.json`, and metadata for Julia's inference pipeline.

## Experimental rules built into the workflow

- Only `train` trips participate in GroupKFold and the Optuna objective.
- A whole trip is kept in one fold.
- Invalid fixes are excluded from fitting and classification metrics.
- Candidate misses count as wrong for Top-1 through the shared evaluator's fix table.
- Median imputation is part of the estimator Pipeline and is fitted inside each fold.
- The feature list is taken from `src.features.FEATURE_COLUMNS`; identifiers and labels are not model inputs.
- `student_test` is a holdout check after fitting/search, not an optimisation target. Evaluator trips are never read.

## Model-selection policy

Primary metric: mean GroupKFold Top-1 accuracy, because it measures the task's per-fix choice of road candidate. Tie-breakers: GroupKFold macro F1, lower GroupKFold log loss (important for HMM emissions), then inference time. The student-test scores are displayed but do not determine the winner. A final selection remains marked provisional until all six classifiers (Logistic Regression, KNN, Gaussian Naive Bayes, Random Forest, XGBoost, MLP) have results.
