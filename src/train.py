"""Train and evaluate one or more registered models (TEAM.md section 2.4).

Run from the project root (needs `python -m src.build_dataset` first):
    python -m src.train --model baseline
    python -m src.train --model logreg --model knn
    python -m src.train --all                     # baseline + every registered model
    python -m src.train --model logreg --quick    # 12 train trips, 3 folds: smoke test only

For each model:
    1. GroupKFold CV on the 80 train trips (whole trips per fold)
    2. refit on all train trips, evaluate once on the 10 student_test trips
    3. write results/models/<name>/{config.json, cv_folds.csv, student_test_metrics.json}
       and replace the model's row in results/models/model_comparison.csv

Evaluator trips are never read.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from . import evaluate as ev
from .config import N_FOLDS, PROCESSED_DIR, RESULTS_DIR, SEED
from .data_io import DATA_DIR
from .features import FEATURE_COLUMNS, load_or_build
from .model_zoo import REGISTRY, get_spec

QUICK_TRIPS = 12
QUICK_FOLDS = 3
CV_COLUMNS = ["macro_f1", "top1", "log_loss", "accuracy", "precision", "recall", "route_continuity",
              "train_macro_f1", "train_log_loss"]


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def load_partition(partition: str, processed_dir: Path, data_dir: Path, rebuild: bool = False):
    fixes_path = Path(processed_dir) / f"fixes_{partition}.parquet"
    if not fixes_path.exists():
        raise FileNotFoundError(f"{fixes_path} not found. Run `python -m src.build_dataset` first.")
    cand = load_or_build(partition, processed_dir, data_dir, rebuild=rebuild)
    fixes = pd.read_parquet(fixes_path)
    return cand, fixes


def run_model(name: str, train: tuple, test: tuple, out_root: Path, n_folds: int, seed: int,
              quick: bool) -> dict:
    cand_tr, fix_tr = train
    cand_te, fix_te = test
    is_baseline = name == "baseline"
    spec = None if is_baseline else get_spec(name)
    family = "Rule (no training)" if is_baseline else spec.family
    description = "nearest candidate arc (cand_rank == 0)" if is_baseline else spec.description
    print(f"\n=== {name} ({family}) ===", flush=True)

    t0 = time.time()
    folds = ev.group_cv(None if is_baseline else spec.factory, {} if is_baseline else spec.params,
                        cand_tr, fix_tr, FEATURE_COLUMNS, n_folds=n_folds,
                        max_train_rows=None if is_baseline else spec.max_train_rows, seed=seed)
    cv_s = time.time() - t0

    if is_baseline:
        st = ev.baseline_metrics(cand_te, fix_te)
        st.update({"train_time_s": 0.0, "infer_ms_per_1k_fixes": float("nan")})
    else:
        model, prob, fit_s, pred_s = ev.fit_predict(spec.factory, spec.params, cand_tr, cand_te, FEATURE_COLUMNS,
                                                    spec.max_train_rows, seed)
        st = ev.evaluate_predictions(cand_te, prob, fix_te)
        st.update(ev.train_fit_metrics(model, cand_tr, FEATURE_COLUMNS, seed))
        st.update({"train_time_s": fit_s, "infer_ms_per_1k_fixes": ev.per_1k_fixes_ms(pred_s, cand_te)})
    print(f"  student_test: macro F1 {st['macro_f1']:.4f}  Top-1 {st['top1']:.4f}  log loss {st['log_loss']:.4f}  "
          f"continuity {st['route_continuity']:.4f}", flush=True)

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": name, "family": family, "description": description,
        "config": {} if is_baseline else spec.config, "features": [] if is_baseline else FEATURE_COLUMNS,
        "seed": seed, "n_folds": n_folds, "quick": quick,
        "train_trips": int(cand_tr["trip_id"].nunique()), "student_test_trips": int(cand_te["trip_id"].nunique()),
        "train_rows_labelled": int(cand_tr["fix_valid"].sum()),
        "cv_seconds": cv_s, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (out_dir / "config.json").write_text(json.dumps(_json_safe(config), indent=2, default=str), encoding="utf-8")
    folds.to_csv(out_dir / "cv_folds.csv", index=False)
    (out_dir / "student_test_metrics.json").write_text(json.dumps(_json_safe(st), indent=2), encoding="utf-8")

    row = {"model": name, "family": family, "key_config": description}
    for c in CV_COLUMNS:
        if c in folds.columns:
            row[f"cv_{c}_mean"] = folds[c].mean()
            row[f"cv_{c}_std"] = folds[c].std(ddof=1) if len(folds) > 1 else 0.0
    for c in ["macro_f1", "top1", "log_loss", "accuracy", "precision", "recall", "route_continuity",
              "candidate_miss_rate", "tn", "fp", "fn", "tp"]:
        row[f"st_{c}"] = st.get(c)
    row["train_time_s"] = st["train_time_s"]
    row["infer_ms_per_1k_fixes"] = st["infer_ms_per_1k_fixes"]
    row["quick"] = quick
    return row


def update_comparison(rows: list[dict], path: Path) -> pd.DataFrame:
    """Replace each model's row (no duplicates when a model is re-run)."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["model"].isin(new["model"])], new], ignore_index=True)
    order = ["baseline"] + [m for m in new["model"] if m != "baseline"]
    new["_o"] = new["model"].map({m: i for i, m in enumerate(dict.fromkeys(order))})
    new = new.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Group-CV + student_test evaluation of registered models")
    ap.add_argument("--model", action="append", default=[], help=f"baseline or one of: {', '.join(REGISTRY)}")
    ap.add_argument("--all", action="store_true", help="baseline + every registered model")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quick", action="store_true", help=f"{QUICK_TRIPS} train trips, {QUICK_FOLDS} folds, "
                                                         "results under results/models/quick/")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "models")
    args = ap.parse_args(argv)

    names = (["baseline"] + list(REGISTRY)) if args.all else args.model
    if not names:
        ap.error("give --model NAME (repeatable) or --all")
    for n in names:
        if n != "baseline":
            get_spec(n)   # fail fast on typos

    train = load_partition("train", args.processed_dir, args.data_dir, args.rebuild_features)
    test = load_partition("student_test", args.processed_dir, args.data_dir, args.rebuild_features)
    n_folds, out_root = args.folds, args.results_dir
    if args.quick:
        keep = sorted(train[0]["trip_id"].unique())[:QUICK_TRIPS]
        train = (train[0][train[0]["trip_id"].isin(keep)], train[1][train[1]["trip_id"].isin(keep)])
        n_folds, out_root = min(n_folds, QUICK_FOLDS), out_root / "quick"
    print(f"train: {train[0]['trip_id'].nunique()} trips, {len(train[0]):,} candidate rows | "
          f"student_test: {test[0]['trip_id'].nunique()} trips, {len(test[0]):,} rows | {n_folds} folds")

    rows = [run_model(n, train, test, out_root, n_folds, args.seed, args.quick) for n in names]
    table = update_comparison(rows, out_root / "model_comparison.csv")
    cols = ["model", "cv_macro_f1_mean", "cv_top1_mean", "st_macro_f1", "st_top1", "st_log_loss",
            "st_route_continuity", "infer_ms_per_1k_fixes"]
    print("\n" + table[[c for c in cols if c in table.columns]].to_string(index=False, float_format="%.4f"))
    print(f"\nwrote {out_root / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
