# XGBoost tuning run

## How it was produced
- Code: branch `joshua/models-tuning-api` at commit `69d50cf` (PR #13), unmodified
- Command: `python -m src.tune --model xgboost --n-trials 40`
- Date: 2026-11-18, 15:42 to 16:08 local time. The Optuna search took 24.2 min.
- Environment: Windows, Python 3.10.4, xgboost 3.2.0, scikit-learn 1.7.2, optuna 4.8.0, pandas 2.3.3, numpy 2.2.6
- Data: `data/train` (80 trips) and `data/student_test` (10 trips) from the repo
  - Tables built with `python -m src.build_dataset` and `python -m src.features`
  - `features.py`, `evaluate.py` and `config.py` are identical on main and this branch
- GroupKFold validation folds: 38,578 / 34,197 / 37,002 / 47,166 / 30,924 labelled fixes
  - These match the folds in this branch's own `random_forest` / `xgboost` `cv_folds.csv`
- Settings: TPE sampler with seed 42, 40 trials, 300,000 training rows per fold during search
  - The objective uses train trips only
  - student_test was evaluated once, after the search
- Random Forest tuning was not run. The guidelines ask for one tuned model, and XGBoost had the higher untuned CV Top-1.

## Tuned vs untuned (`tuned_vs_untuned.csv`)

| Metric | Untuned (defaults) | Tuned (best trial) |
|---|---|---|
| CV Top-1 (objective) | 84.35% ± 6.6 | 84.40% ± 6.4 |
| CV macro F1 | 0.9149 | 0.9008 |
| CV log loss | 0.0774 | 0.1020 |
| Student-test Top-1 | 87.30% | 87.31% |
| Student-test macro F1 | 0.9267 | 0.9044 |
| Student-test log loss | 0.0605 | 0.0887 |
| Inference ms per 1,000 fixes | 12.5 | 7.7 |

## Best parameters (`best_params.json`)

| Parameter | Tuned | Default |
|---|---|---|
| `n_estimators` | 150 | 350 |
| `learning_rate` | 0.1151 | 0.06 |
| `max_depth` | 8 | 6 |
| `min_child_weight` | 9.0191 | 3.0 |
| `subsample` | 0.9085 | 0.85 |
| `colsample_bytree` | 0.6296 | 0.85 |
| `reg_lambda` | 0.1525 | 2.0 |
| `gamma` | 0.5793 | - |
| `scale_pos_weight` | 6.6358 | 1.0 |

## Note
- `tuning_summary.json`: the `outputs` paths were changed from absolute local paths to repo-relative paths. Nothing else was edited.
