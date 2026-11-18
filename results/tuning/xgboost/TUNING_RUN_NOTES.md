# XGBoost tuning run 2

## How it was produced
- Command: `python -m src.tune --model xgboost --n-trials 40`
- Change from run 1: `scale_pos_weight` is fixed at 1 instead of searched.
  - Run 1 (`../xgboost_run1_with_scale_pos_weight/`) picked 6.6, which left Top-1 unchanged but raised student-test log loss from 0.060 to 0.089.
  - Up-weighting the positive class inflates the probabilities that the HMM uses as emissions.
- Search: Optuna TPE, seed 42, 40 trials, objective group_cv_top1 on the 80 train trips only.
  - 300,000 training rows per fold during the search.
  - student_test was evaluated once, after the search.
- Search time: 7.4 min.
- Environment: macOS, Python 3.12.15, xgboost 3.2.0, scikit-learn 1.7.2, optuna 4.8.0.

## Tuned vs untuned (`tuned_vs_untuned.csv`)

| Metric | Untuned | Tuned |
|---|---|---|
| CV Top-1 (objective) | 84.69% ± 4.3% | 84.78% ± 4.0% |
| CV macro F1 | 0.9168 | 0.9181 |
| CV log loss | 0.0768 | 0.0756 |
| Student-test Top-1 | 87.49% | 87.08% |
| Student-test macro F1 | 0.9273 | 0.9262 |
| Student-test log loss | 0.0602 | 0.0609 |
| Inference ms per 1,000 fixes | 4.8 | 6.0 |

## Reading
- The tuned configuration is slightly better on all three CV metrics, but the gains are far smaller than the fold-to-fold spread.
- On student_test it is slightly lower on Top-1. Tuning gives no meaningful improvement: the default configuration was already close to optimal for these features.
- The model is selected on CV, not on student_test, so the tuned model is exported to `artifacts/model.joblib`. The two are effectively equivalent.
- Run 1 vs run 2 shows that searching `scale_pos_weight` harmed probability quality without improving accuracy.

## Best parameters (`best_params.json`)
```json
{
  "n_estimators": 400,
  "learning_rate": 0.022867628178107367,
  "max_depth": 6,
  "min_child_weight": 3.275997058271572,
  "subsample": 0.7226377748197736,
  "colsample_bytree": 0.7337248481829695,
  "reg_lambda": 0.9601967316825686,
  "gamma": 2.198642888110843
}
```
