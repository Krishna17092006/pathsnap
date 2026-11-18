"""Optuna/TPE tuning for PathSnap's Random Forest and XGBoost models.

This module deliberately delegates cross-validation and metrics to the team's
``src.evaluate.group_cv`` implementation. Thus the tuned models use exactly
the same GroupKFold split and metric definitions as the ordinary training run.
Only train trips enter the Optuna objective. Student-test trips are evaluated
once after search and never influence parameter selection.

Examples:
    python -m src.build_dataset
    python -m src.features --partition all
    python -m src.train --model xgboost
    python -m src.tune --model xgboost --n-trials 40
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from .config import ARTIFACTS_DIR, N_FOLDS, PROCESSED_DIR, RESULTS_DIR, SEED
from .evaluate import evaluate_predictions, group_cv, training_rows
from .features import FEATURE_COLUMNS, load_or_build, save_feature_list
from .model_zoo import MODEL_ZOO, canonical_model_name

DEFAULT_TRIAL_TRAIN_ROWS = 300_000


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _numeric_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix = frame.loc[:, columns].copy()
    for column in columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)


def _load_feature_tables(force: bool = False):
    """Use the canonical feature builder/cache from the feature-engineering owner."""
    train_candidates = load_or_build("train", rebuild=force)
    student_candidates = load_or_build("student_test", rebuild=force)
    train_fixes_path = PROCESSED_DIR / "fixes_train.parquet"
    student_fixes_path = PROCESSED_DIR / "fixes_student_test.parquet"
    if not train_fixes_path.exists() or not student_fixes_path.exists():
        raise FileNotFoundError("Fix parquet files missing; run `python -m src.build_dataset` first.")
    train_fixes = pd.read_parquet(train_fixes_path)
    student_fixes = pd.read_parquet(student_fixes_path)

    for label, frame in (("train", train_candidates), ("student_test", student_candidates)):
        missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{label} feature table missing FEATURE_COLUMNS: {missing}")
    train_ids = set(train_candidates["trip_id"].astype(str).unique())
    test_ids = set(student_candidates["trip_id"].astype(str).unique())
    if train_ids & test_ids:
        raise ValueError("Train and student_test trip IDs overlap; refusing to tune with leakage")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_feature_list(ARTIFACTS_DIR / "features.json")
    return train_candidates, train_fixes, student_candidates, student_fixes


def _cv_summary(
    model_name: str,
    candidates: pd.DataFrame,
    fixes: pd.DataFrame,
    params: dict[str, Any],
    max_train_rows: int | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Call the project's shared evaluator; don't maintain a parallel CV definition."""
    _, factory, _ = MODEL_ZOO[model_name]
    folds = group_cv(
        factory=factory,
        params=params,
        cand=candidates,
        fixes=fixes,
        features=FEATURE_COLUMNS,
        n_folds=N_FOLDS,
        max_train_rows=max_train_rows,
        seed=SEED,
        verbose=False,
    )
    def mean_std(column: str) -> tuple[float | None, float | None]:
        if column not in folds:
            return None, None
        values = pd.to_numeric(folds[column], errors="coerce").dropna()
        if values.empty:
            return None, None
        return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

    summary: dict[str, Any] = {}
    for source, dest in (
        ("top1", "top1"),
        ("macro_f1", "macro_f1"),
        ("log_loss", "log_loss"),
        ("candidate_miss_rate", "candidate_miss_rate"),
        ("route_continuity", "route_continuity"),
        ("infer_ms_per_1k_fixes", "infer_ms_per_1k_fixes"),
        ("train_time_s", "train_time_s"),
    ):
        avg, std = mean_std(source)
        summary[f"{dest}_mean"] = avg
        summary[f"{dest}_std"] = std
    summary["folds"] = folds.to_dict(orient="records")
    summary["max_train_rows_per_fold"] = max_train_rows
    return summary, folds


def _fit_and_score_holdout(
    model_name: str,
    train_candidates: pd.DataFrame,
    student_candidates: pd.DataFrame,
    student_fixes: pd.DataFrame,
    params: dict[str, Any],
):
    _, factory, _ = MODEL_ZOO[model_name]
    rows = training_rows(train_candidates, max_rows=None, seed=SEED)
    if rows["label"].nunique() < 2:
        raise ValueError("Training data must include both positive and negative candidate labels")
    model = factory(**params)
    x_train = _numeric_matrix(rows, FEATURE_COLUMNS)
    start = time.perf_counter()
    model.fit(x_train, rows["label"].astype(int).to_numpy())
    fit_seconds = time.perf_counter() - start

    x_test = _numeric_matrix(student_candidates, FEATURE_COLUMNS)
    start = time.perf_counter()
    probabilities = model.predict_proba(x_test)[:, 1] if len(x_test) else np.array([], dtype=float)
    predict_seconds = time.perf_counter() - start
    metrics = evaluate_predictions(student_candidates, probabilities, student_fixes)
    n_fixes = len(student_fixes)
    metrics["inference_ms_per_1000_fixes"] = predict_seconds * 1_000_000.0 / n_fixes if n_fixes else float("nan")
    metrics["fit_seconds"] = float(fit_seconds)
    metrics["predict_seconds"] = float(predict_seconds)
    metrics["n_candidate_rows"] = int(len(student_candidates))
    return model, metrics


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            # scale_pos_weight is fixed at 1 (not searched). Run 1 searched it and picked 6.6: Top-1 was
            # unchanged but log loss rose from 0.060 to 0.089, because up-weighting the positive class
            # inflates the probabilities that the HMM uses as emissions.
            # See results/tuning/xgboost_run1_with_scale_pos_weight/.
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [12, 16, 20, 24, 28, 32, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
            "max_samples": trial.suggest_float("max_samples", 0.5, 0.95),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced_subsample"]),
        }
    raise ValueError(f"No search space defined for {model_name!r}")


def _rel(path: Path) -> str:
    """Project-relative path for result files (no local absolute paths in the repo)."""
    try:
        return path.resolve().relative_to(RESULTS_DIR.parent.resolve()).as_posix()
    except ValueError:
        return path.name


def _trial_frame(study: optuna.Study) -> pd.DataFrame:
    records = []
    for trial in study.trials:
        record = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "duration_seconds": trial.duration.total_seconds() if trial.duration else np.nan,
        }
        record.update({f"param_{key}": value for key, value in trial.params.items()})
        record.update(trial.user_attrs)
        records.append(record)
    return pd.DataFrame(records)


def _save_plots(study: optuna.Study, out_dir: Path, objective_metric: str) -> None:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not complete:
        return
    xs = [t.number for t in complete]
    ys = [float(t.value) for t in complete]
    best = np.maximum.accumulate(ys)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3, label="Trial objective")
    ax.plot(xs, best, linestyle="--", linewidth=1.4, label="Best so far")
    ax.set_xlabel("Optuna trial")
    metric_label = "Top-1 accuracy" if objective_metric == "top1" else "Macro F1"
    ax.set_ylabel(f"Group-CV objective ({metric_label})")
    ax.set_title("Optuna optimisation history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "optimization_history.png", dpi=160)
    plt.close(fig)

    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as exc:
        print(f"Parameter importance skipped: {exc}")
        return
    if not importance:
        return
    items = sorted(importance.items(), key=lambda pair: pair[1])
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.38 * len(items))))
    ax.barh([item[0] for item in items], [item[1] for item in items])
    ax.set_xlabel("Relative importance")
    ax.set_title("Optuna parameter importance")
    fig.tight_layout()
    fig.savefig(out_dir / "parameter_importance.png", dpi=160)
    plt.close(fig)


def run_tuning(
    model_name: str,
    n_trials: int = 40,
    objective_metric: str = "top1",
    force_features: bool = False,
    max_train_rows: int | None = DEFAULT_TRIAL_TRAIN_ROWS,
) -> dict[str, Any]:
    canonical = canonical_model_name(model_name)
    if canonical not in {"random_forest", "xgboost"}:
        raise ValueError("Only Random Forest and XGBoost are tuned here")
    if objective_metric not in {"top1", "macro_f1"}:
        raise ValueError("objective_metric must be 'top1' or 'macro_f1'")
    if n_trials < 1:
        raise ValueError("n_trials must be at least one")
    if max_train_rows is not None and max_train_rows < 1:
        raise ValueError("max_train_rows must be positive or None")

    train_candidates, train_fixes, student_candidates, student_fixes = _load_feature_tables(force_features)
    out_dir = RESULTS_DIR / "tuning" / canonical
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "optuna_study.sqlite3"
    storage = f"sqlite:///{db_path.resolve().as_posix()}"
    study = optuna.create_study(
        study_name=f"pathsnap_{canonical}_{objective_metric}",
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        direction="maximize",
        load_if_exists=True,
    )

    def persist_trials(current_study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        _trial_frame(current_study).to_csv(out_dir / "optuna_trials.csv", index=False)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, canonical)
        summary, _ = _cv_summary(canonical, train_candidates, train_fixes, params, max_train_rows)
        for key, value in summary.items():
            if key != "folds" and value is not None and np.isfinite(float(value)):
                trial.set_user_attr(key, float(value))
        return float(summary[f"{objective_metric}_mean"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search_start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, callbacks=[persist_trials], n_jobs=1)
    search_seconds = time.perf_counter() - search_start
    _trial_frame(study).to_csv(out_dir / "optuna_trials.csv", index=False)
    _save_plots(study, out_dir, objective_metric)

    _, _, defaults = MODEL_ZOO[canonical]
    best_params = dict(study.best_params)
    base_cv, base_folds = _cv_summary(canonical, train_candidates, train_fixes, defaults, max_train_rows)
    tuned_cv, tuned_folds = _cv_summary(canonical, train_candidates, train_fixes, best_params, max_train_rows)
    base_folds.to_csv(out_dir / "untuned_cv_folds.csv", index=False)
    tuned_folds.to_csv(out_dir / "tuned_cv_folds.csv", index=False)

    base_model, base_test = _fit_and_score_holdout(
        canonical, train_candidates, student_candidates, student_fixes, defaults
    )
    tuned_model, tuned_test = _fit_and_score_holdout(
        canonical, train_candidates, student_candidates, student_fixes, best_params
    )
    joblib.dump(tuned_model, out_dir / "tuned_model.joblib", compress=3)

    rows = []
    for label, cv, test in (("untuned", base_cv, base_test), ("tuned", tuned_cv, tuned_test)):
        rows.append({
            "model": canonical,
            "configuration": label,
            "cv_top1_mean": cv["top1_mean"],
            "cv_top1_std": cv["top1_std"],
            "cv_macro_f1_mean": cv["macro_f1_mean"],
            "cv_macro_f1_std": cv["macro_f1_std"],
            "cv_log_loss_mean": cv["log_loss_mean"],
            "cv_candidate_miss_rate_mean": cv["candidate_miss_rate_mean"],
            "student_test_top1": test.get("top1"),
            "student_test_macro_f1": test.get("macro_f1"),
            "student_test_log_loss": test.get("log_loss"),
            "student_test_candidate_miss_rate": test.get("candidate_miss_rate"),
            "inference_ms_per_1000_fixes": test.get("inference_ms_per_1000_fixes"),
            "fit_seconds": test.get("fit_seconds"),
            "n_valid_student_fixes": test.get("n_valid_fixes"),
            "cv_max_train_rows_per_fold": max_train_rows,
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "tuned_vs_untuned.csv", index=False)

    best_params_doc = {
        "model": canonical,
        "objective": f"group_cv_{objective_metric}",
        "best_value": float(study.best_value),
        "best_params": best_params,
        "default_params": defaults,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "n_trials_total": len(study.trials),
        "n_trials_this_run": n_trials,
        "max_train_rows_per_fold_during_search": max_train_rows,
        "search_seconds": search_seconds,
        "final_fit_seconds": tuned_test["fit_seconds"],
        "feature_columns": FEATURE_COLUMNS,
        "note": "Optuna objective uses train trips only; student_test is holdout-only. Final fitted estimator uses all valid train candidate rows.",
    }
    (out_dir / "best_params.json").write_text(json.dumps(_json_safe(best_params_doc), indent=2), encoding="utf-8")
    summary = {
        "model": canonical,
        "objective": f"group_cv_{objective_metric}",
        "best_params": best_params,
        "best_value": float(study.best_value),
        "search_seconds": search_seconds,
        "n_trials_total": len(study.trials),
        "max_train_rows_per_fold_during_search": max_train_rows,
        "outputs": {
            "trials": _rel(out_dir / "optuna_trials.csv"),
            "history_plot": _rel(out_dir / "optimization_history.png"),
            "importance_plot": _rel(out_dir / "parameter_importance.png"),
            "tuned_model": _rel(out_dir / "tuned_model.joblib"),
            "comparison": _rel(out_dir / "tuned_vs_untuned.csv"),
        },
    }
    (out_dir / "tuning_summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    print("\nTuned vs untuned:")
    print(comparison.to_string(index=False))
    print(f"\nOptuna search finished in {search_seconds / 60:.1f} min; best {objective_metric}={study.best_value:.5f}")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Tune a PathSnap tree model using shared GroupKFold metrics and Optuna TPE")
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "xgb", "random_forest", "rf"])
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--objective", choices=["top1", "macro_f1"], default="top1")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument(
        "--max-train-rows", type=int, default=DEFAULT_TRIAL_TRAIN_ROWS,
        help=f"maximum rows sampled from each CV training fold during HPO (default: {DEFAULT_TRIAL_TRAIN_ROWS}); final refit still uses all valid train rows",
    )
    parser.add_argument(
        "--full-fold-data", action="store_true",
        help="use all candidate rows in every Optuna CV fit (slower); overrides --max-train-rows",
    )
    args = parser.parse_args(argv)
    if args.n_trials < 1:
        parser.error("--n-trials must be at least one")
    cap = None if args.full_fold_data else args.max_train_rows
    run_tuning(args.model, args.n_trials, args.objective, args.force_features, cap)


if __name__ == "__main__":
    main()
