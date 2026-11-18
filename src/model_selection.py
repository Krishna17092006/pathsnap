"""Evidence-based final model selection from the shared comparison CSV.

Selection policy:
1. Primary: mean GroupKFold Top-1 accuracy (directly measures the per-fix
   decision PathSnap must make).
2. Tie-break 1: mean GroupKFold macro F1 (candidate classification quality).
3. Tie-break 2: mean GroupKFold log loss (probability quality for HMM emissions).
4. Tie-break 3: inference latency per 1,000 fixes (deployment practicality).

Student-test metrics are reported as a holdout sanity check, not used to rank
models. This avoids selecting on the development holdout and leaves evaluator
trips completely untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS_DIR

ALIASES = {
    "model": ("model", "name", "model_name"),
    "cv_top1": ("cv_top1_mean", "cv_top1_accuracy_mean", "cv_top1_acc_mean", "cv_top1", "group_cv_top1", "cv_top1_accuracy", "top1_mean"),
    "cv_macro_f1": ("cv_macro_f1_mean", "cv_macro_f1", "group_cv_macro_f1"),
    "cv_log_loss": ("cv_log_loss_mean", "cv_logloss_mean", "cv_log_loss", "group_cv_log_loss", "log_loss_mean"),
    "inference_ms_1000": (
        "infer_ms_per_1k_fixes", "inference_ms_per_1000_fixes", "inference_time_ms_per_1000_fixes",
        "inference_ms_1000_fixes", "inference_ms",
    ),
    "student_top1": ("st_top1", "student_test_top1", "student_top1", "test_top1"),
    "student_macro_f1": ("st_macro_f1", "student_test_macro_f1", "student_macro_f1", "test_macro_f1"),
    "student_log_loss": ("st_log_loss", "student_test_log_loss", "student_log_loss", "test_log_loss"),
}


def _find_column(columns, aliases):
    lookup = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def rank_models(table: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """Return ranked rows, selected primary metric, and missing metric fields."""
    if table.empty:
        raise ValueError("Model comparison table is empty")
    cols = {key: _find_column(table.columns, names) for key, names in ALIASES.items()}
    if cols["model"] is None:
        raise ValueError("Comparison CSV needs a model/name column")

    # Prefer Top-1 as the primary metric. Older/shared comparison writers may
    # initially report only macro F1; use it explicitly as the fallback.
    primary = "cv_top1" if cols["cv_top1"] else "cv_macro_f1"
    if cols[primary] is None:
        raise ValueError("Need at least cv_top1 or cv_macro_f1 to select a model")

    out = table.copy()
    out["_model_name"] = out[cols["model"]].astype(str)
    sort_cols, ascending = [], []
    primary_col = cols[primary]
    out["_primary_score"] = pd.to_numeric(out[primary_col], errors="coerce")
    sort_cols.append("_primary_score")
    ascending.append(False)

    secondary = "cv_macro_f1" if primary != "cv_macro_f1" else "cv_top1"
    if cols[secondary]:
        out["_secondary_score"] = pd.to_numeric(out[cols[secondary]], errors="coerce")
        sort_cols.append("_secondary_score")
        ascending.append(False)

    if cols["cv_log_loss"]:
        out["_cv_log_loss"] = pd.to_numeric(out[cols["cv_log_loss"]], errors="coerce")
        sort_cols.append("_cv_log_loss")
        ascending.append(True)
    if cols["inference_ms_1000"]:
        out["_inference_ms_1000"] = pd.to_numeric(out[cols["inference_ms_1000"]], errors="coerce")
        sort_cols.append("_inference_ms_1000")
        ascending.append(True)

    ranked = out.sort_values(sort_cols, ascending=ascending, na_position="last", kind="stable")
    missing = [key for key in ("cv_top1", "cv_macro_f1", "cv_log_loss", "inference_ms_1000") if cols[key] is None]
    return ranked.drop(columns=[c for c in ranked if c.startswith("_")]), primary, missing


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Select a PathSnap model using GroupKFold results")
    parser.add_argument("--comparison", type=Path, default=RESULTS_DIR / "models" / "model_comparison.csv")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "models")
    args = parser.parse_args(argv)

    if not args.comparison.exists():
        raise FileNotFoundError(
            f"{args.comparison} does not exist yet. Run the shared model training/evaluation workflow first."
        )
    table = pd.read_csv(args.comparison)
    ranked, primary, missing = rank_models(table)
    cols = {key: _find_column(ranked.columns, names) for key, names in ALIASES.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.out_dir / "model_selection_ranking.csv", index=False)

    winner = ranked.iloc[0].to_dict()
    # Do not call a winner final until the comparison contains every team's
    # six classifiers. The nearest-arc baseline is a reference row, not one of
    # the six learned models.
    name_col = _find_column(ranked.columns, ALIASES["model"])
    names = {str(v).strip().lower().replace("-", "_").replace(" ", "_") for v in ranked[name_col]}
    expected_groups = {
        "logistic_regression": {"logistic_regression", "logreg", "lr"},
        "knn": {"knn", "k_nearest_neighbors", "k_nearest_neighbours"},
        "gaussian_naive_bayes": {"gaussian_naive_bayes", "gaussian_nb", "naive_bayes", "gnb"},
        "random_forest": {"random_forest", "randomforest", "rf"},
        "xgboost": {"xgboost", "xgb"},
        "mlp": {"mlp", "multilayer_perceptron"},
    }
    missing_team_models = [
        label for label, aliases in expected_groups.items()
        if not any(name in aliases for name in names)
    ]
    report = {
        "selected_model": str(winner[name_col]),
        "selected_primary_score": (float(winner[cols[primary]]) if pd.notna(winner[cols[primary]]) else None),
        "selected_cv_macro_f1": (float(winner[cols["cv_macro_f1"]]) if cols["cv_macro_f1"] and pd.notna(winner[cols["cv_macro_f1"]]) else None),
        "selected_cv_log_loss": (float(winner[cols["cv_log_loss"]]) if cols["cv_log_loss"] and pd.notna(winner[cols["cv_log_loss"]]) else None),
        "selected_inference_ms_per_1000_fixes": (float(winner[cols["inference_ms_1000"]]) if cols["inference_ms_1000"] and pd.notna(winner[cols["inference_ms_1000"]]) else None),
        "primary_metric": primary,
        "selection_policy": [
            "GroupKFold mean Top-1 accuracy (or macro F1 when Top-1 is unavailable)",
            "GroupKFold macro F1 / Top-1 as the secondary criterion",
            "lower GroupKFold log loss",
            "lower inference milliseconds per 1,000 fixes",
        ],
        "comparison_rows": int(len(ranked)),
        "missing_metrics": missing,
        "missing_required_tree_models": missing_team_models,
        "selection_status": "provisional" if missing_team_models else "ready_for_team_review",
        "student_test_metrics_are_not_used_to_rank": True,
    }
    (args.out_dir / "model_selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Ranked {len(ranked)} model rows using {primary!r} as the primary metric.")
    print(ranked.head(10).to_string(index=False))
    print(f"Selection: {report['selected_model']} ({report['selection_status']})")
    if missing:
        print(f"Missing comparison metrics: {', '.join(missing)}")
    if missing_team_models:
        print(f"Missing model entries before finalising: {', '.join(missing_team_models)}")


if __name__ == "__main__":
    main()
