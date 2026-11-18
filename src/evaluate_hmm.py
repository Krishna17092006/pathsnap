"""Compare classifier-only predictions with classifier + HMM on student_test.

Run after selecting/saving a classifier artifact and tuning `artifacts/hmm.json`:
    python -m src.evaluate_hmm            # all 10 student_test trips
    python -m src.evaluate_hmm --quick    # first 2 trips (smoke test)

Outputs go to one folder per emission model (the model in artifacts/, e.g. xgboost or mlp),
like results/models/<model>/:

    results/hmm/<model>/
    ├── classifier_vs_hmm.csv        one row per trip: Top-1, continuity, route P/R/F1, timings
    ├── classifier_vs_hmm.png        mean Top-1 and route continuity, classifier vs HMM
    ├── hmm_non_improvements.csv     trips where HMM Top-1 <= classifier Top-1
    ├── integration_failures.csv     trips where predict_trip raised (empty when all succeed)
    ├── integration_summary.json     means, win counts, timings and decoder settings of this run
    └── predictions.parquet          per-fix trip_id, fix_idx, arc_clf, arc_hmm
    maps/results/<model>_hmm/        maps of the two trips where the HMM helped least

predictions.parquet is the input format of `python -m src.result_maps --predictions`.
"""
from __future__ import annotations

import json
import time
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .data_io import list_trips, load_trip
from .labels import make_labels
from .maps import result_map
from .pipeline import load_artifacts, predict_trip, score_trip
from .result_maps import classify

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _summary(result: pd.DataFrame, failures: list, art: dict) -> dict:
    means = result[["top1_classifier", "top1_hmm", "route_continuity_clf", "route_continuity_hmm",
                    "classifier_route_f1", "hmm_route_f1"]].mean()
    model_path = Path(art["model_path"]).name if art.get("model_path") else None
    return {
        "partition": "student_test",
        "emission_model": art["model_name"],
        "model_file": f"artifacts/{model_path}" if model_path else None,
        "decoder": art["decoder_config"].to_dict(),
        "n_student_test_trips": int(len(result)),
        "n_integration_failures": len(failures),
        "student_test_means": {k: float(v) for k, v in means.items()},
        "trips_improved_by_hmm": {
            "top1": int((result.top1_hmm > result.top1_classifier).sum()),
            "route_continuity": int((result.route_continuity_hmm > result.route_continuity_clf).sum()),
            "route_f1": int((result.hmm_route_f1 > result.classifier_route_f1).sum()),
        },
        "mean_inference_ms": float(result.inference_ms.mean()),
        "max_inference_ms": float(result.inference_ms.max()),
        "note": "predict_trip ran with trip.route = None; the reference route was used only by score_trip. "
                "Timings depend on the machine.",
    }


def _failure_map(trip, pred, row: dict, model_name: str):
    """Result map (src.maps.result_map) of one trip, with per-fix error types as in src.result_maps."""
    track = trip.track
    ok = (np.isfinite(track["lon"]) & np.isfinite(track["lat"]) & np.isfinite(track["t"])
          & track["lat"].between(-90, 90) & track["lon"].between(-180, 180)).to_numpy()
    net = pred._network
    fx = pred.fixes.iloc[np.flatnonzero(ok)].reset_index(drop=True).copy()
    px, py = net.proj.to_xy(fx["lon"].to_numpy(), fx["lat"].to_numpy())
    labels = make_labels(px, py, net, trip.route)
    fx["err_clf"] = classify(fx["arc_clf"], labels.true_arc, labels.fix_valid, net)
    fx["err_hmm"] = classify(fx["arc_hmm"], labels.true_arc, labels.fix_valid, net)
    title = (f"Trip {trip.trip_id} · classifier ({model_name}) Top-1 {row['top1_classifier'] * 100:.1f}% · "
             f"HMM Top-1 {row['top1_hmm'] * 100:.1f}%")
    return result_map(trip, net, fx, title=title, clf_name=f"Classifier-only ({model_name})")


def run(artifact_dir=None, quick=False):
    art = load_artifacts(artifact_dir) if artifact_dir else load_artifacts()
    name = art["model_name"]
    trips = list_trips("student_test")
    if quick:
        trips = trips[:2]
    rows, failures, map_cases, predictions = [], [], [], []
    out_dir = RESULTS_DIR / "hmm" / name
    map_dir = PROJECT_ROOT / "maps" / "results" / f"{name}_hmm"
    out_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    model_file = Path(art["model_path"]).name if art.get("model_path") else None
    print(f"Emission model: {name} (artifacts/{model_file}); decoder: {art['decoder_config'].to_dict()}",
          flush=True)
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
            predictions.append(pred.fixes[["fix_idx", "arc_clf", "arc_hmm"]].assign(trip_id=tid))
            if metrics["top1_hmm"] <= metrics["top1_classifier"]:
                map_cases.append((trip, pred))
            print(f"{tid}: classifier Top-1={metrics['top1_classifier']:.4f}, "
                  f"HMM Top-1={metrics['top1_hmm']:.4f}, time={elapsed_ms:.0f} ms", flush=True)
        except Exception as exc:
            failures.append({"trip_id": tid, "error": repr(exc)})
            print(f"{tid}: FAILED: {exc}", flush=True)
    # Maps are required for examples where temporal decoding did not help. Two
    # contrasting cases are enough for the report; all are listed in hmm_non_improvements.csv.
    map_cases.sort(key=lambda item: next(
        (r["top1_hmm"] - r["top1_classifier"] for r in rows if r["trip_id"] == item[0].trip_id), 0.0))
    for trip, pred in map_cases[:2]:
        row = next(r for r in rows if r["trip_id"] == trip.trip_id)
        _failure_map(trip, pred, row, name).save(str(map_dir / f"{trip.trip_id}_hmm_no_improvement.html"))
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "classifier_vs_hmm.csv", index=False)
    non_improve = result[result.top1_hmm <= result.top1_classifier].copy() if len(result) else result
    non_improve.to_csv(out_dir / "hmm_non_improvements.csv", index=False)
    pd.DataFrame(failures, columns=["trip_id", "error"]).to_csv(out_dir / "integration_failures.csv", index=False)
    if predictions:
        (pd.concat(predictions, ignore_index=True)[["trip_id", "fix_idx", "arc_clf", "arc_hmm"]]
         .to_parquet(out_dir / "predictions.parquet", index=False))
    if len(result):
        summary = _summary(result, failures, art)
        (out_dir / "integration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        means = result[["top1_classifier", "top1_hmm", "route_continuity_clf", "route_continuity_hmm"]].mean()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(["Classifier", "HMM"], [means.top1_classifier, means.top1_hmm])
        axes[0].set_title("Student-test Top-1 accuracy")
        axes[1].bar(["Classifier", "HMM"], [means.route_continuity_clf, means.route_continuity_hmm])
        axes[1].set_title("Student-test route continuity")
        for ax in axes:
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Emissions from {name}")
        fig.tight_layout()
        fig.savefig(out_dir / "classifier_vs_hmm.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    print(f"\nSaved comparison to {out_dir.relative_to(PROJECT_ROOT).as_posix()}/")
    print(f"Result maps: {map_dir.relative_to(PROJECT_ROOT).as_posix()}/")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(args.artifact_dir, args.quick)
