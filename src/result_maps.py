"""Result maps and error breakdown: predictions vs the reference route on student_test trips.

Run from the project root (after `python -m src.build_dataset`):
    python -m src.result_maps
        Nearest-candidate baseline (classifier-only stand-in until the real pipeline exists).
    python -m src.result_maps --predictions results/predictions/<name>_student_test.parquet --name <name>
        Any method's per-fix predictions.

Predictions file: one row per fix with columns
    trip_id (str), fix_idx (int), arc_clf (int, -1 = none) and optionally arc_hmm (int, -1 = none).

Writes
    maps/results/<name>/<trip_id>.html                  interactive map per trip
    results/result_maps/<name>_error_breakdown.csv      per-trip error types
    results/result_maps/<name>_summary.md
Errors are counted on labelled fixes only (fix_valid). A prediction is correct when it equals the
reference arc or an exact duplicate (same u, v), the same rule as the candidate labels.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, PROJECT_ROOT, RESULTS_DIR
from .data_io import load_trip
from .geo import LocalProjection
from .maps import ERROR_TEXT, result_map
from .network import RoadNetwork

PARTITION = "student_test"
ERROR_TYPES = ["correct", "wrong_direction", "wrong_road", "no_prediction"]


def classify(pred, true, valid, net: RoadNetwork) -> np.ndarray:
    """Per-fix outcome: correct / wrong_direction / wrong_road / no_prediction (None where not labelled)."""
    pred = np.asarray(pred, dtype="int64")
    true = np.asarray(true, dtype="int64")
    out = np.full(len(pred), None, dtype=object)
    lab = np.asarray(valid, dtype=bool) & (true >= 0)
    p, t = np.where(pred >= 0, pred, 0), np.where(true >= 0, true, 0)
    same = (net.u[p] == net.u[t]) & (net.v[p] == net.v[t])
    rev = (net.u[p] == net.v[t]) & (net.v[p] == net.u[t])
    out[lab & (pred < 0)] = "no_prediction"
    out[lab & (pred >= 0) & same] = "correct"
    out[lab & (pred >= 0) & ~same & rev] = "wrong_direction"
    out[lab & (pred >= 0) & ~same & ~rev] = "wrong_road"
    return out


def baseline_predictions() -> pd.DataFrame:
    """Nearest candidate per fix (cand_rank 0); fixes without candidates get -1."""
    cand = pd.read_parquet(PROCESSED_DIR / f"candidates_{PARTITION}.parquet",
                           columns=["trip_id", "fix_idx", "arc_id", "cand_rank"])
    near = cand[cand["cand_rank"] == 0][["trip_id", "fix_idx", "arc_id"]].rename(columns={"arc_id": "arc_clf"})
    fixes = pd.read_parquet(PROCESSED_DIR / f"fixes_{PARTITION}.parquet", columns=["trip_id", "fix_idx"])
    out = fixes.merge(near, on=["trip_id", "fix_idx"], how="left")
    out["arc_clf"] = out["arc_clf"].fillna(-1).astype("int64")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Result maps and error breakdown on student_test")
    ap.add_argument("--predictions", help="parquet with trip_id, fix_idx, arc_clf[, arc_hmm]")
    ap.add_argument("--name", default=None, help="name used for output folders (default: baseline)")
    args = ap.parse_args(argv)

    name = args.name or ("baseline" if not args.predictions else "predictions")
    preds = baseline_predictions() if not args.predictions else pd.read_parquet(args.predictions)
    preds["trip_id"] = preds["trip_id"].astype(str)
    methods = [c for c in ("arc_clf", "arc_hmm") if c in preds]
    label = {"arc_clf": "classifier" if args.predictions else "nearest-candidate baseline", "arc_hmm": "HMM"}

    fixes_all = pd.read_parquet(PROCESSED_DIR / f"fixes_{PARTITION}.parquet")
    map_dir = PROJECT_ROOT / "maps" / "results" / name
    out_dir = RESULTS_DIR / "result_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for tid, fx in fixes_all.groupby("trip_id", sort=True):
        fx = fx.sort_values("fix_idx").merge(preds[["trip_id", "fix_idx"] + methods], on=["trip_id", "fix_idx"], how="left")
        for c in methods:
            fx[c] = fx[c].fillna(-1).astype("int64")
        trip = load_trip(tid, PARTITION)
        proj = LocalProjection.from_points(fx["lon"].to_numpy(), fx["lat"].to_numpy())
        net = RoadNetwork(trip.nodes, trip.arcs, proj)
        parts = []
        for c in methods:
            err_col = "err_" + c.split("_")[1]
            fx[err_col] = classify(fx[c], fx["true_arc"], fx["fix_valid"], net)
            counts = pd.Series(fx[err_col]).value_counts()
            n_lab = int(fx["fix_valid"].sum())
            row = {"trip_id": tid, "method": label[c], "labelled_fixes": n_lab}
            row.update({k: int(counts.get(k, 0)) for k in ERROR_TYPES})
            row["top1"] = row["correct"] / n_lab if n_lab else np.nan
            rows.append(row)
            parts.append(f"{label[c]} Top-1 {row['top1'] * 100:.1f}%")
        m = result_map(trip, net, fx, title=f"Trip {tid} · " + " · ".join(parts),
                       clf_name="Classifier-only" if args.predictions else "Nearest-road baseline")
        m.save(str(map_dir / f"{tid}.html"))
        print(f"[{tid}] {len(fx):>5,} fixes | " + " | ".join(parts), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{name}_error_breakdown.csv", index=False)
    tot = df.groupby("method", sort=False)[["labelled_fixes"] + ERROR_TYPES].sum()
    lines = [f"# Result maps: {name}", "",
             f"Generated by `python -m src.result_maps{'' if not args.predictions else ' --predictions ...'}` "
             f"on the {PARTITION} partition. Errors are counted on labelled fixes.", "",
             "| Method | Labelled fixes | Correct | Right road, wrong direction | Wrong road | No prediction |",
             "|---|---|---|---|---|---|"]
    for meth, r in tot.iterrows():
        n = r["labelled_fixes"]
        lines.append(f"| {meth} | {n:,} | " + " | ".join(f"{r[k]:,} ({r[k] / n * 100:.1f}%)" for k in ERROR_TYPES) + " |")
    lines += ["", f"Per-trip numbers: `results/result_maps/{name}_error_breakdown.csv`. "
              f"Maps: `maps/results/{name}/<trip_id>.html`.", ""]
    (out_dir / f"{name}_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[4:4 + len(tot) + 2]))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
