"""Tune HMM decoder parameters on TRAIN trips only.

This script builds the small candidate/fix tables in memory, so it does not
require the Parquet cache to exist. It never reads student_test or evaluator
trips. By default it uses a deterministic, size-stratified subset of 3 train
trips with at most 1,500 fixes each; pass --max-trips 80 to use every
training trip (this can be substantially slower).

Run:
    python -m src.tune_hmm
    python -m src.tune_hmm --quick
    python -m src.tune_hmm --max-trips 20
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from types import SimpleNamespace

from .candidates import build_tables
from .config import CANDIDATE_RADIUS_M, RESULTS_DIR, SEED
from .data_io import load_trip, list_trips
from .geo import LocalProjection
from .features import add_features
from .hmm import DecoderConfig, viterbi_decode
from .labels import make_labels
from .network import RoadNetwork
from .pipeline import _model_probabilities, load_artifacts
from .route_metrics import pointwise_accuracy, route_continuity, sequence_metrics


def _prepare_trip(trip_id: str, art: dict):
    trip = load_trip(trip_id, "train")
    tr = trip.track
    valid = (np.isfinite(tr.lon) & np.isfinite(tr.lat) & np.isfinite(tr.t)
             & tr.lat.between(-90, 90) & tr.lon.between(-180, 180)).to_numpy()
    raw_indices = np.flatnonzero(valid)
    valid_track = tr.iloc[raw_indices].reset_index(drop=True)
    if valid_track.empty:
        raise ValueError(f"Trip {trip_id} has no valid GPS fixes")
    proj = LocalProjection.from_points(valid_track.lon.to_numpy(), valid_track.lat.to_numpy())
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    px, py = proj.to_xy(valid_track.lon.to_numpy(), valid_track.lat.to_numpy())
    labels = make_labels(px, py, net, trip.route)
    cand, fixes = build_tables(trip, net, px, py, labels, CANDIDATE_RADIUS_M,
                               fix_idx_map=raw_indices)
    prob = None
    if art.get("model_bundle") is not None and not cand.empty:
        # Feature engineering uses compact valid-fix indices, then probabilities
        # are kept aligned with the original candidate row order.
        local_cand = cand.copy()
        local_cand["fix_idx"] = np.searchsorted(raw_indices, local_cand.fix_idx.to_numpy(dtype="int64"))
        feature_trip = SimpleNamespace(track=valid_track, trip_id=trip.trip_id,
                                        partition=trip.partition)
        featured = add_features(local_cand, feature_trip, net)
        prob = _model_probabilities(featured, art)
    return trip, net, cand, fixes, prob


def _evaluate_one(trip, net, cand: pd.DataFrame, fixes: pd.DataFrame,
                  probabilities, config: DecoderConfig):
    pred_hmm = viterbi_decode(cand, fixes, net, probabilities=probabilities, config=config)
    # Classifier-only prediction = highest-probability candidate per fix (ties -> nearer candidate).
    # Without a model it falls back to the nearest candidate, which is then the baseline.
    clf = np.full(len(fixes), -1, dtype="int64")
    fix_pos = {int(fid): i for i, fid in enumerate(fixes.fix_idx)}
    scored = cand.assign(_p=probabilities if probabilities is not None else -cand["cand_rank"].to_numpy())
    top = scored.sort_values(["fix_idx", "_p", "cand_rank"], ascending=[True, False, True],
                             kind="stable").drop_duplicates("fix_idx")
    for row in top.itertuples():
        clf[fix_pos[int(row.fix_idx)]] = int(row.arc_id)
    m_clf = pointwise_accuracy(fixes, clf, net)
    m_hmm = pointwise_accuracy(fixes, pred_hmm, net)
    c_clf = route_continuity(fixes, clf, net, config.gap_threshold_s)
    c_hmm = route_continuity(fixes, pred_hmm, net, config.gap_threshold_s)
    seq = sequence_metrics([x for x in pred_hmm if x >= 0], trip.route, net)
    return {
        "trip_id": trip.trip_id,
        "top1_classifier": m_clf["top1"], "top1_hmm": m_hmm["top1"],
        "route_continuity_classifier": c_clf, "route_continuity_hmm": c_hmm,
        "route_precision": seq["route_precision"], "route_recall": seq["route_recall"],
        "route_f1": seq["route_f1"], "n_valid_fixes": m_hmm["n_valid_fixes"],
    }


def _select_trip_ids(max_trips: int, quick: bool = False) -> list[str]:
    ids = list_trips("train")
    if max_trips >= len(ids):
        return ids
    sizes = [(tid, len(load_trip(tid, "train").track)) for tid in ids]
    sizes.sort(key=lambda x: x[1])
    if quick:
        # A fast smoke test should not include the 19k-fix stress-test trip.
        return [tid for tid, _ in sizes[:max_trips]]
    # Keep the default tuning pass tractable and size-stratified. The full
    # 19k-fix trip remains covered by the decoder stress/integration test.
    eligible = [(tid, n) for tid, n in sizes if n <= 1_500]
    if len(eligible) < max_trips:
        eligible = sizes
    indices = np.linspace(0, len(eligible) - 1, max_trips).round().astype(int)
    return [eligible[i][0] for i in indices]


DEFAULT_BETAS = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0)
DEFAULT_WEIGHTS = (0.5, 1.0, 2.0)
DEFAULT_GAPS = (30.0, 60.0)


def run(quick: bool = False, max_trips: int = 3, betas=DEFAULT_BETAS, weights=DEFAULT_WEIGHTS,
        gaps=DEFAULT_GAPS):
    trip_ids = _select_trip_ids(2 if quick else max_trips, quick=quick)
    art = load_artifacts()
    emission_source = "classifier_probability" if art.get("model_bundle") is not None else "distance_gaussian"
    print(f"Tuning on {len(trip_ids)} TRAIN trips only: {trip_ids}; emissions={emission_source}", flush=True)
    prepared = []
    for i, tid in enumerate(trip_ids, 1):
        started = time.perf_counter()
        item = _prepare_trip(tid, art)
        prepared.append(item)
        print(f"  prepared {i}/{len(trip_ids)} {tid}: {len(item[2]):,} candidates, "
              f"{len(item[3]):,} fixes ({time.perf_counter() - started:.1f}s)", flush=True)

    # Sigma only affects the distance-Gaussian fallback. With classifier
    # probabilities it is held at the saved default rather than duplicating
    # identical grid rows.
    sigmas = [5.0, 10.0] if emission_source == "distance_gaussian" else [8.0]
    combos = list(itertools.product([float(b) for b in betas], sigmas, [float(g) for g in gaps],
                                    [float(w) for w in weights]))
    rows = []
    start = time.perf_counter()
    for k, (beta, sigma, gap, weight) in enumerate(combos, 1):
        cfg = DecoderConfig(beta=beta, sigma=sigma, gap_threshold_s=gap,
                            emission_weight=weight)
        trip_rows = []
        for trip, net, cand, fixes, prob in prepared:
            trip_rows.append(_evaluate_one(trip, net, cand, fixes, prob, cfg))
        d = pd.DataFrame(trip_rows)
        rows.append({
            "beta": beta, "sigma": sigma, "gap_threshold_s": gap,
            "emission_weight": weight, "emission_source": emission_source,
            "mean_top1_classifier": d.top1_classifier.mean(),
            "mean_top1_hmm": d.top1_hmm.mean(),
            "mean_route_continuity_classifier": d.route_continuity_classifier.mean(),
            "mean_route_continuity_hmm": d.route_continuity_hmm.mean(),
            "mean_route_f1": d.route_f1.mean(),
            "objective": 0.7 * d.top1_hmm.mean() + 0.3 * d.route_continuity_hmm.mean(),
            "n_trips": len(d),
        })
        print(f"[{k}/{len(combos)}] beta={beta:g}, sigma={sigma:g}, gap={gap:g}, "
              f"weight={weight:g} -> Top-1={rows[-1]['mean_top1_hmm']:.4f}", flush=True)

    out = pd.DataFrame(rows).sort_values(
        ["objective", "mean_top1_hmm", "mean_route_continuity_hmm"],
        ascending=False).reset_index(drop=True)
    # One folder per emission model, like results/models/<model>/.
    out_dir = RESULTS_DIR / "hmm" / art["model_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "decoder_grid.csv", index=False)
    best = out.iloc[0]
    best_cfg = DecoderConfig(beta=float(best.beta), sigma=float(best.sigma),
                             gap_threshold_s=float(best.gap_threshold_s),
                             emission_weight=float(best.emission_weight))
    with (out_dir / "best_decoder.json").open("w", encoding="utf-8") as f:
        json.dump(best_cfg.to_dict(), f, indent=2)
    summary = {"partition": "train", "trip_ids": trip_ids, "n_trips": len(trip_ids),
               "n_configurations": len(combos), "seed": SEED,
               "emission_source": emission_source, "emission_model": art["model_name"],
               "best_config": best_cfg.to_dict(),
               "grid": {"betas": list(betas), "emission_weights": list(weights), "gap_threshold_s": list(gaps)},
               "classifier_baseline": "highest classifier probability per fix (nearest candidate without a model)",
               "selection_note": ("Two smallest training trips for a smoke test." if quick else
                                  "Deterministic size-stratified subset of TRAIN trips with <=1500 fixes; "
                                  "use --max-trips 80 for the full training partition.")}
    (out_dir / "tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # This is Julia's decoder artifact; it does not touch Joshua's classifier model.
    art_dir = Path(__file__).resolve().parents[1] / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "hmm.json").write_text(json.dumps(best_cfg.to_dict(), indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    scatter = ax.scatter(out.mean_top1_hmm, out.mean_route_continuity_hmm,
                         c=out.beta, s=45, cmap="viridis")
    ax.set_xlabel("Mean train Top-1 accuracy")
    ax.set_ylabel("Mean train route continuity")
    ax.set_title("HMM decoder grid (TRAIN trips only)")
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="beta (m)")
    fig.tight_layout()
    fig.savefig(out_dir / "decoder_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nBest configuration: {best_cfg.to_dict()}")
    print(f"Elapsed: {time.perf_counter() - start:.1f}s")
    print(f"Saved: {out_dir / 'decoder_grid.csv'}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use 2 training trips for a smoke test")
    parser.add_argument("--max-trips", type=int, default=3,
                        help="Number of size-stratified TRAIN trips (use 80 for all)")
    parser.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--gaps", type=float, nargs="+", default=list(DEFAULT_GAPS))
    args = parser.parse_args()
    run(args.quick, args.max_trips, args.betas, args.weights, args.gaps)
