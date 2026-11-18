"""Train/evaluate a registered PathSnap model using the shared evaluator.

This entry point is intentionally compatible with src.features.load_or_build
and src.evaluate.group_cv, rather than introducing a second evaluation system.

Examples (from the repository root):
    python -m src.build_dataset
    python -m src.features --partition all
    python -m src.train --model random_forest
    python -m src.train --model xgboost
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR, N_FOLDS, PROCESSED_DIR, RESULTS_DIR, SEED
from .evaluate import evaluate_predictions, group_cv, training_rows
from .features import FEATURE_COLUMNS, load_or_build, save_feature_list
from .model_zoo import MODEL_ZOO, canonical_model_name


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


def _mean_std(frame: pd.DataFrame, column: str) -> tuple[float | None, float | None]:
    if column not in frame:
        return None, None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None, None
    return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else 0.0)


def train_one(model_name: str, rebuild_features: bool = False, n_folds: int = N_FOLDS,
              max_train_rows: int | None = None) -> dict[str, Any]:
    canonical = canonical_model_name(model_name)
    family, factory, defaults = MODEL_ZOO[canonical]

    # Feature implementation is owned by the team feature-engineering module.
    # Use its cache builder and its canonical ordered feature list.
    train_candidates = load_or_build("train", rebuild=rebuild_features)
    student_candidates = load_or_build("student_test", rebuild=rebuild_features)
    train_fixes_path = PROCESSED_DIR / "fixes_train.parquet"
    student_fixes_path = PROCESSED_DIR / "fixes_student_test.parquet"
    for required in (train_fixes_path, student_fixes_path):
        if not required.exists():
            raise FileNotFoundError(f"{required} missing. Run `python -m src.build_dataset` first.")
    train_fixes = pd.read_parquet(train_fixes_path)
    student_fixes = pd.read_parquet(student_fixes_path)

    missing_train = sorted(set(FEATURE_COLUMNS) - set(train_candidates.columns))
    missing_test = sorted(set(FEATURE_COLUMNS) - set(student_candidates.columns))
    if missing_train or missing_test:
        raise ValueError(f"Feature table mismatch: train missing={missing_train}; student_test missing={missing_test}")
    if not set(train_candidates["trip_id"].astype(str)).isdisjoint(set(student_candidates["trip_id"].astype(str))):
        raise ValueError("Train and student_test trip IDs overlap; refusing to evaluate with leakage")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_feature_list(ARTIFACTS_DIR / "features.json")

    out_dir = RESULTS_DIR / "models" / canonical
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{canonical}] GroupKFold({n_folds}) over {train_candidates['trip_id'].nunique()} training trips")
    cv_folds = group_cv(
        factory=factory,
        params=defaults,
        cand=train_candidates,
        fixes=train_fixes,
        features=FEATURE_COLUMNS,
        n_folds=n_folds,
        max_train_rows=max_train_rows,
        seed=SEED,
        verbose=True,
    )
    cv_folds.to_csv(out_dir / "cv_folds.csv", index=False)

    cv_top1_mean, cv_top1_std = _mean_std(cv_folds, "top1")
    cv_f1_mean, cv_f1_std = _mean_std(cv_folds, "macro_f1")
    cv_loss_mean, cv_loss_std = _mean_std(cv_folds, "log_loss")
    cv_miss_mean, _ = _mean_std(cv_folds, "candidate_miss_rate")
    cv_infer_mean, _ = _mean_std(cv_folds, "infer_ms_per_1k_fixes")

    valid_train = training_rows(train_candidates, max_rows=max_train_rows, seed=SEED)
    if valid_train["label"].nunique() < 2:
        raise ValueError("Training table needs both positive and negative candidate labels")
    x_train = _numeric_matrix(valid_train, FEATURE_COLUMNS)
    model = factory(**defaults)
    t0 = time.perf_counter()
    model.fit(x_train, valid_train["label"].astype(int).to_numpy())
    fit_seconds = time.perf_counter() - t0

    x_test = _numeric_matrix(student_candidates, FEATURE_COLUMNS)
    t0 = time.perf_counter()
    test_prob = model.predict_proba(x_test)[:, 1] if len(x_test) else np.array([], dtype=float)
    predict_seconds = time.perf_counter() - t0
    student_metrics = evaluate_predictions(student_candidates, test_prob, student_fixes)
    student_n_fixes = int(len(student_fixes))
    student_metrics["inference_ms_per_1000_fixes"] = (
        predict_seconds * 1_000_000.0 / student_n_fixes if student_n_fixes else float("nan")
    )
    student_metrics["fit_seconds"] = float(fit_seconds)
    student_metrics["predict_seconds"] = float(predict_seconds)
    student_metrics["n_candidate_rows"] = int(len(student_candidates))
    student_metrics["n_valid_train_candidate_rows"] = int(len(valid_train))
    (out_dir / "student_test_metrics.json").write_text(
        json.dumps(_json_safe(student_metrics), indent=2), encoding="utf-8"
    )

    config = {
        "model": canonical,
        "family": family,
        "params": defaults,
        "seed": SEED,
        "n_folds": n_folds,
        "feature_columns": FEATURE_COLUMNS,
        "train_candidate_rows": int(len(valid_train)),
        "max_train_rows_per_fold_and_final_fit": max_train_rows,
        "train_trips": int(train_candidates["trip_id"].nunique()),
        "student_test_trips": int(student_candidates["trip_id"].nunique()),
    }
    (out_dir / "config.json").write_text(json.dumps(_json_safe(config), indent=2), encoding="utf-8")
    joblib.dump(model, out_dir / "model.joblib", compress=3)

    comparison_row = {
        "model": canonical,
        "family": family,
        "cv_top1_mean": cv_top1_mean,
        "cv_top1_std": cv_top1_std,
        "cv_macro_f1_mean": cv_f1_mean,
        "cv_macro_f1_std": cv_f1_std,
        "cv_log_loss_mean": cv_loss_mean,
        "cv_log_loss_std": cv_loss_std,
        "cv_candidate_miss_rate_mean": cv_miss_mean,
        "inference_ms_per_1000_fixes": cv_infer_mean,
        "student_test_top1": student_metrics.get("top1"),
        "student_test_macro_f1": student_metrics.get("macro_f1"),
        "student_test_log_loss": student_metrics.get("log_loss"),
        "student_test_candidate_miss_rate": student_metrics.get("candidate_miss_rate"),
        "student_test_inference_ms_per_1000_fixes": student_metrics.get("inference_ms_per_1000_fixes"),
        "fit_seconds": fit_seconds,
        "params": json.dumps(_json_safe(defaults), sort_keys=True),
    }
    comparison_path = RESULTS_DIR / "models" / "model_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        if "model" in comparison.columns:
            comparison = comparison[comparison["model"].astype(str) != canonical]
        comparison = pd.concat([comparison, pd.DataFrame([comparison_row])], ignore_index=True, sort=False)
    else:
        comparison = pd.DataFrame([comparison_row])
    comparison.to_csv(comparison_path, index=False)

    print(f"\n{canonical} complete")
    print(f"  Group-CV Top-1:   {cv_top1_mean:.4f} ± {cv_top1_std:.4f}")
    print(f"  Group-CV macro F1:{cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
    print(f"  Group-CV log loss:{cv_loss_mean:.4f} ± {cv_loss_std:.4f}")
    print(f"  Student-test Top-1: {student_metrics['top1']:.4f}")
    print(f"  Outputs: {out_dir}")
    return {"model": canonical, "cv": cv_folds, "student_test": student_metrics, "comparison_row": comparison_row}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a PathSnap model using GroupKFold")
    parser.add_argument("--model", required=True, help=f"Model name. Registered: {', '.join(sorted(MODEL_ZOO))}")
    parser.add_argument("--rebuild-features", action="store_true", help="rebuild cached train and student-test feature tables")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--max-train-rows", type=int, default=None,
                        help="optional stratified candidate-row cap for each CV fit and the final fit; default trains on all valid rows")
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.max_train_rows is not None and args.max_train_rows < 1:
        parser.error("--max-train-rows must be positive")
    train_one(args.model, rebuild_features=args.rebuild_features, n_folds=args.folds,
              max_train_rows=args.max_train_rows)


if __name__ == "__main__":
    main()
