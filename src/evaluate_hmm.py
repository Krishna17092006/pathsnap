"""Compare classifier-only predictions with classifier + HMM on student_test.

Run after selecting/saving a classifier artifact and tuning `artifacts/hmm.json`:
    python -m src.evaluate_hmm
"""
from __future__ import annotations

import time
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .data_io import list_trips, load_trip
from .maps import result_map
from .pipeline import load_artifacts, predict_trip, score_trip


def run(artifact_dir=None, quick=False):
    art = load_artifacts(artifact_dir) if artifact_dir else load_artifacts()
    trips = list_trips("student_test")
    if quick:
        trips = trips[:2]
    rows, failures, map_cases = [], [], []
    map_dir = Path(__file__).resolve().parents[1] / "maps" / "results"
    map_dir.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / "hmm"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tid in trips:
        trip = load_trip(tid, "student_test")
        t0 = time.perf_counter()
        try:
            # Explicitly remove the reference route from the inference object.
            # Ground truth is only passed later to score_trip on the server side.
            inference_trip = copy(trip)
            inference_trip.route = None
            pred = predict_trip(inference_trip, art)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            metrics = score_trip(pred, trip)
            rows.append({"trip_id": tid, **metrics, "inference_ms": elapsed_ms,
                         "n_candidates": pred.diagnostics.get("n_candidates", 0),
                         "n_unmatched_fixes": pred.diagnostics.get("n_unmatched_fixes", 0),
                         "candidate_generation_ms": pred.diagnostics.get("candidate_generation_ms"),
                         "model_ms": pred.diagnostics.get("model_ms"),
                         "hmm_ms": pred.diagnostics.get("hmm_ms")})
            if metrics["top1_hmm"] <= metrics["top1_classifier"]:
                map_cases.append((trip, pred))
            pd.DataFrame(rows).to_csv(out_dir / "classifier_vs_hmm.csv", index=False)
            pd.DataFrame(failures).to_csv(out_dir / "integration_failures.csv", index=False)
            print(f"{tid}: classifier Top-1={metrics['top1_classifier']:.4f}, "
                  f"HMM Top-1={metrics['top1_hmm']:.4f}, time={elapsed_ms:.0f} ms", flush=True)
        except Exception as exc:
            failures.append({"trip_id": tid, "error": repr(exc)})
            print(f"{tid}: FAILED: {exc}", flush=True)
    # Maps are required for examples where temporal decoding did not help; do
    # not spend time serialising a large map for every successful trip.
    # Two contrasting failure examples are enough for the report; all cases
    # remain listed in hmm_non_improvements.csv. Avoid generating ten huge maps.
    map_cases.sort(key=lambda item: next(
        (r["top1_hmm"] - r["top1_classifier"] for r in rows if r["trip_id"] == item[0].trip_id), 0.0))
    for trip, pred in map_cases[:2]:
        result_map(trip, pred._network, pred).save(map_dir / f"{trip.trip_id}_hmm_no_improvement.html")
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "classifier_vs_hmm.csv", index=False)
    non_improve = result[result.top1_hmm <= result.top1_classifier].copy() if len(result) else result
    non_improve.to_csv(out_dir / "hmm_non_improvements.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "integration_failures.csv", index=False)
    if len(result):
        means = result[["top1_classifier", "top1_hmm", "route_continuity_clf", "route_continuity_hmm"]].mean()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(["Classifier", "HMM"], [means.top1_classifier, means.top1_hmm])
        axes[0].set_title("Student-test Top-1 accuracy")
        axes[1].bar(["Classifier", "HMM"], [means.route_continuity_clf, means.route_continuity_hmm])
        axes[1].set_title("Student-test route continuity")
        for ax in axes:
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "classifier_vs_hmm.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    print(f"\nSaved comparison to {out_dir}")
    print(f"Result maps: {map_dir}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.artifact_dir, args.quick)
