"""Build per-fix labels and the shared candidate table (train + student_test only).

Run from the project root:
    python -m src.build_dataset            # all trips, radius from src/config.py
    python -m src.build_dataset --quick    # 3 trips per partition, written to .../quick/

Writes
    data/processed/candidates_<partition>.parquet   one row per (fix, candidate arc)   [not in git]
    data/processed/fixes_<partition>.parquet        one row per GPS fix                [not in git]
    results/candidates/miss_rate.csv                miss rate and table size vs search radius
    results/candidates/miss_rate_per_trip.csv       per-trip numbers at the chosen radius
    results/candidates/label_stats.csv              per-trip label coverage and checks
    results/candidates/figures/*.png
    results/candidates/dataset_summary.md           data decisions and the numbers behind them
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plot_style as ps
from .candidates import build_tables, candidate_pairs
from .config import CANDIDATE_RADIUS_M, LABEL_COST_CAP_M, LABEL_MAX_DIST_M, PROCESSED_DIR, RESULTS_DIR
from .data_io import PARTITIONS, list_trips, load_trip
from .geo import LocalProjection
from .labels import make_labels
from .network import RoadNetwork

RADII_M = (25, 50, 75, 100)
FAST_SPEED_MS = 8.0          # ~29 km/h: above this, GPS heading is reliable enough to check label direction
SLOW_SPEED_MS = 2.5          # ~9 km/h: below this, heading is dominated by jitter
LOW_QUALITY_OPPOSITE = 0.05  # trips where > 5% of fast fixes oppose their label direction are flagged


def process_trip(trip_id: str, partition: str, radius_m: float):
    trip = load_trip(trip_id, partition)
    tr = trip.track
    ok = (np.isfinite(tr["lon"]) & np.isfinite(tr["lat"]) & np.isfinite(tr["t"])
          & (tr["lat"].abs() <= 90) & (tr["lon"].abs() <= 180)).to_numpy()
    idx_map = np.flatnonzero(ok)
    lon, lat = tr["lon"].to_numpy()[ok], tr["lat"].to_numpy()[ok]
    proj = LocalProjection.from_points(lon, lat)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    px, py = proj.to_xy(lon, lat)
    labels = make_labels(px, py, net, trip.route)

    key = {"partition": partition, "trip_id": trip_id}
    valid = labels.fix_valid

    # ---- radius analysis on the widest radius, then filter
    pairs = candidate_pairs(net, px, py, max(RADII_M), labels)
    radius_rows = []
    for rad in RADII_M:
        sub = pairs[pairs["dist_m"] <= rad]
        fix = sub["fix_idx"].to_numpy()
        n_c = np.bincount(fix, minlength=len(px))
        pos_fix = np.zeros(len(px), dtype=bool)
        pos_fix[fix[sub["label"].to_numpy() == 1]] = True
        nearest = sub[sub["cand_rank"] == 0]
        top1 = np.zeros(len(px), dtype=bool)
        top1[nearest["fix_idx"].to_numpy()[nearest["label"].to_numpy() == 1]] = True
        radius_rows.append({**key, "radius_m": rad, "n_fixes": len(px), "valid_fixes": int(valid.sum()),
                            "misses": int((valid & ~pos_fix).sum()), "fixes_without_candidates": int((n_c == 0).sum()),
                            "rows": int(len(sub)), "cand_sum": int(n_c.sum()),
                            "cand_median": float(np.median(n_c)), "cand_p95": float(np.percentile(n_c, 95)),
                            "cand_max": int(n_c.max()) if len(n_c) else 0,
                            "nearest_top1_hits": int((valid & top1).sum())})

    cand, fixes = build_tables(trip, net, px, py, labels, radius_m, fix_idx_map=idx_map, pairs=pairs)

    # ---- label checks
    stats = dict(key)
    stats.update({"n_fixes": len(px), "valid_fixes": int(valid.sum()), "share_valid": float(valid.mean()) if len(px) else 0.0})
    if trip.has_route:
        route = np.asarray(trip.route)
        assert np.all(np.diff(labels.route_pos) >= 0), "route positions must never decrease"
        d_near, a_near = net.distance_to_arcs(px, py, np.unique(route))
        ta = np.where(valid, labels.true_arc, 0)
        same = (net.u[a_near] == net.u[ta]) & (net.v[a_near] == net.v[ta])
        # Direction check: at driving speed the labelled arc should point the way the vehicle travels.
        # Heading = displacement from fix i-1 to i+1; it is only meaningful when the vehicle moves fast
        # enough for GPS jitter to be small relative to the displacement.
        n = len(px)
        t_v = trip.track["t"].to_numpy()[ok]
        hx, hy, dt2 = np.zeros(n), np.zeros(n), np.ones(n)
        if n > 2:
            hx[1:-1], hy[1:-1], dt2[1:-1] = px[2:] - px[:-2], py[2:] - py[:-2], t_v[2:] - t_v[:-2]
        speed = np.hypot(hx, hy) / np.maximum(dt2, 1e-9)
        ax_, ay_ = net.bx[ta] - net.ax[ta], net.by[ta] - net.ay[ta]
        arc_len = np.hypot(ax_, ay_)
        cosang = (hx * ax_ + hy * ay_) / np.maximum(np.hypot(hx, hy) * arc_len, 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        fast = valid & (speed > FAST_SPEED_MS) & (arc_len > 1.0)
        stats.update({
            "fast_fixes": int(fast.sum()),
            "heading_agrees_with_label": float((ang[fast] <= 90).mean()) if fast.any() else np.nan,
            "opposite_direction_share": float((ang[fast] > 135).mean()) if fast.any() else np.nan,
            "slow_share": float((speed[valid] <= SLOW_SPEED_MS).mean()) if valid.any() else np.nan,
            "route_arcs": len(route),
            "route_positions_covered": float(len(np.unique(labels.route_pos[valid])) / len(route)) if valid.any() else 0.0,
            "match_dist_median_m": float(np.median(labels.match_dist_m[valid])) if valid.any() else np.nan,
            "match_dist_p95_m": float(np.percentile(labels.match_dist_m[valid], 95)) if valid.any() else np.nan,
            "extra_dist_vs_nearest_route_arc_m": float(np.mean((labels.match_dist_m - d_near)[valid])) if valid.any() else np.nan,
            "agrees_with_nearest_route_arc": float(same[valid].mean()) if valid.any() else np.nan,
        })
    return cand, fixes, radius_rows, stats


def aggregate_radius(rows: pd.DataFrame) -> pd.DataFrame:
    g = rows.groupby(["partition", "radius_m"], sort=False).sum(numeric_only=True)
    out = pd.DataFrame({
        "trips": rows.groupby(["partition", "radius_m"], sort=False)["trip_id"].nunique(),
        "fixes": g["n_fixes"], "valid_fixes": g["valid_fixes"], "misses": g["misses"],
        "miss_rate": g["misses"] / g["valid_fixes"],
        "fixes_without_candidates": g["fixes_without_candidates"],
        "mean_candidates_per_fix": g["cand_sum"] / g["n_fixes"],
        "candidate_rows": g["rows"],
        "nearest_candidate_top1": g["nearest_top1_hits"] / g["valid_fixes"],
    }).reset_index()
    return out


def make_figures(agg: pd.DataFrame, label_stats: pd.DataFrame, chosen: float, fig_dir) -> list:
    ps.apply_style()
    out = []
    parts = [p for p in PARTITIONS if (agg["partition"] == p).any()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for part in parts:
        d = agg[agg["partition"] == part]
        c = ps.PARTITION_COLOR[part]
        axes[0].plot(d["radius_m"], d["miss_rate"] * 100, marker="o", ms=6, color=c, label=ps.PARTITION_LABEL[part])
        axes[1].plot(d["radius_m"], d["mean_candidates_per_fix"], marker="o", ms=6, color=c, label=ps.PARTITION_LABEL[part])
        for x, y in zip(d["radius_m"], d["miss_rate"] * 100):
            axes[0].annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 7), ha="center",
                             fontsize=7.5, color=ps.INK_2)
    for ax in axes:
        ax.axvline(chosen, color=ps.BASELINE, ls="--", lw=1, zorder=1)
        ax.set_xticks(list(RADII_M))
        ax.set_xlabel("Candidate search radius, m")
        ax.legend()
    axes[0].set_ylabel("Candidate miss rate, % of labelled fixes")
    axes[0].set_title("True arc missing from the candidate set")
    axes[1].set_ylabel("Mean candidate arcs per fix")
    axes[1].set_title("Candidate-set size")
    axes[0].text(chosen, axes[0].get_ylim()[1], f" chosen {chosen:.0f} m", color=ps.MUTED, fontsize=8, va="top")
    out.append(ps.save(fig, fig_dir / "fig_miss_rate_vs_radius.png"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9))
    rng = np.random.default_rng(0)
    for ax, col, xlabel in [(axes[0], "share_valid", "% of fixes with a reliable label"),
                            (axes[1], "opposite_direction_share",
                             f"% of fast fixes (> {FAST_SPEED_MS:.0f} m/s) moving against their labelled arc")]:
        for row, part in enumerate(parts):
            v = label_stats.loc[label_stats["partition"] == part, col].dropna().to_numpy() * 100
            ax.scatter(v, row + rng.uniform(-0.17, 0.17, len(v)), s=28, color=ps.PARTITION_COLOR[part],
                       edgecolors=ps.SURFACE, linewidths=0.8, zorder=3)
            if len(v):
                ax.plot([np.median(v)] * 2, [row - 0.33, row + 0.33], color=ps.INK, lw=1.8, zorder=4)
        ax.set_yticks(range(len(parts)), [ps.PARTITION_LABEL[p] for p in parts])
        ax.set_ylim(len(parts) - 0.45, -0.55)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[1].set_yticklabels([])
    fig.suptitle("Label coverage per trip", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.03, "Each dot is one trip; the black tick marks the partition median.", fontsize=8, color=ps.MUTED)
    out.append(ps.save(fig, fig_dir / "fig_label_coverage.png"))
    return out


def write_summary(agg, per_trip, label_stats, chosen, sizes, figs, out_dir, elapsed) -> None:
    def pct(x):
        return f"{x * 100:.2f}%"

    L = ["# Dataset build: labels and candidates", "",
         "Generated by `python -m src.build_dataset`. Train and student_test only; evaluator data is never read.", "",
         "## Data decisions", "",
         f"1. **Labels by order-constrained route matching.** Each fix gets a position along the reference route through a "
         f"monotone dynamic programme (position never moves backwards), so U-turns, loops and revisited arcs are resolved "
         f"by time order instead of plain proximity. Matching cost is distance capped at {LABEL_COST_CAP_M:.0f} m.",
         f"2. **Unreliable labels are dropped, not guessed.** A fix whose matched arc is more than {LABEL_MAX_DIST_M:.0f} m away is "
         f"marked `fix_valid = False`: the reference route does not describe that part of the drive (EDA: trips 00000054, "
         f"00000033). These fixes are excluded from training and metrics but kept for HMM inference.",
         f"3. **Label threshold > candidate radius.** Labels use {LABEL_MAX_DIST_M:.0f} m while candidates use {chosen:.0f} m, "
         f"so the candidate miss rate is a real measurement and not 0% by construction.",
         "4. **Duplicate arcs count as correct.** Some networks contain arcs with identical start and end nodes; a candidate "
         "that duplicates the true arc gets `label = 1`, because no model can separate identical geometry.",
         f"5. **Candidate radius = {chosen:.0f} m**, chosen from the miss-rate / table-size trade-off below.",
         "6. **Fixes with no candidate within the radius** have no rows in the candidate table; they are listed in the fix "
         "table (`n_cand = 0`) so metrics and the HMM can account for them.", "",
         "## Candidate radius analysis", "",
         "| Partition | Radius (m) | Labelled fixes | Miss rate | Fixes without candidates | Mean candidates / fix | Candidate rows | Nearest-candidate Top-1 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in agg.itertuples():
        L.append(f"| {ps.PARTITION_LABEL[r.partition]} | {r.radius_m} | {r.valid_fixes:,} | {pct(r.miss_rate)} | "
                 f"{r.fixes_without_candidates:,} | {r.mean_candidates_per_fix:.1f} | {r.candidate_rows:,} | {pct(r.nearest_candidate_top1)} |")
    L += ["", f"The miss rate at {LABEL_MAX_DIST_M:.0f} m is 0% by construction (labels require the matched arc within "
          f"{LABEL_MAX_DIST_M:.0f} m), so only the smaller radii are informative.", "",
          "*Nearest-candidate Top-1* is the naive baseline: pick the geometrically nearest arc. On two-way roads both "
          "directions are equally near, so this baseline often picks the wrong direction; heading features address that.", "",
          "## Labels", ""]
    for part in [p for p in PARTITIONS if (label_stats["partition"] == p).any()]:
        s = label_stats[label_stats["partition"] == part]
        L.append(f"- {ps.PARTITION_LABEL[part]}: {int(s.valid_fixes.sum()):,} of {int(s.n_fixes.sum()):,} fixes labelled "
                 f"({s.valid_fixes.sum() / s.n_fixes.sum() * 100:.2f}%); median match distance {s.match_dist_median_m.median():.1f} m; "
                 f"label equals the plain nearest route arc for {s.agrees_with_nearest_route_arc.median() * 100:.1f}% of fixes "
                 f"(median trip). Lowest coverage: "
                 + ", ".join(f"{r.trip_id} ({r.share_valid * 100:.1f}%)" for r in s.nsmallest(3, "share_valid").itertuples()) + ".")
        L.append(f"  - Direction check (fixes faster than {FAST_SPEED_MS:.0f} m/s): labelled arc points the way the vehicle "
                 f"moves for {s.heading_agrees_with_label.median() * 100:.1f}% of fixes (median trip); opposite direction "
                 f"(> 135 deg) for {s.opposite_direction_share.median() * 100:.2f}% (median trip).")
    flagged = label_stats[label_stats["opposite_direction_share"] > LOW_QUALITY_OPPOSITE]
    L += ["", "### Low label-quality trips", "",
          f"Trips where more than {LOW_QUALITY_OPPOSITE * 100:.0f}% of fast fixes move against their labelled arc. Labels "
          "there are noisy (GPS jitter or a reference route that does not match the drive). They are kept in the tables; "
          "whether to exclude them from training should be decided by a cross-validation experiment.", ""]
    if len(flagged):
        L.append("| Partition | Trip | Labelled share | Opposite-direction share (fast fixes) | Slow-fix share |")
        L.append("|---|---|---|---|---|")
        for r in flagged.sort_values("opposite_direction_share", ascending=False).itertuples():
            L.append(f"| {ps.PARTITION_LABEL[r.partition]} | {r.trip_id} | {r.share_valid * 100:.1f}% | "
                     f"{r.opposite_direction_share * 100:.1f}% | {r.slow_share * 100:.1f}% |")
    else:
        L.append("None.")
    L += ["", "## Output tables", ""]
    for name, (rows, mb) in sizes.items():
        L.append(f"- `data/processed/{name}`: {rows:,} rows, {mb:.1f} MB")
    L += ["", "Figures:"] + [f"- `{p.relative_to(RESULTS_DIR.parent).as_posix()}`" for p in figs]
    L += ["", f"Runtime: {elapsed:.0f} s.", ""]
    (out_dir / "dataset_summary.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build labels and the candidate table")
    ap.add_argument("--radius", type=float, default=CANDIDATE_RADIUS_M, help="candidate search radius in metres")
    ap.add_argument("--quick", action="store_true", help="3 trips per partition, written to a quick/ subfolder")
    args = ap.parse_args(argv)

    t0 = time.time()
    proc_dir = PROCESSED_DIR / "quick" if args.quick else PROCESSED_DIR
    out_dir = RESULTS_DIR / "candidates" / ("quick" if args.quick else "")
    fig_dir = out_dir / "figures"
    proc_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    radius_rows, label_rows, per_trip_rows, sizes = [], [], [], {}
    for part in PARTITIONS:
        trips = list_trips(part)[:3] if args.quick else list_trips(part)
        cands, fixes = [], []
        for k, tid in enumerate(trips, 1):
            ts = time.time()
            cand, fx, rrows, stats = process_trip(tid, part, args.radius)
            cands.append(cand)
            fixes.append(fx)
            radius_rows += rrows
            label_rows.append(stats)
            chosen = next(r for r in rrows if r["radius_m"] == min(RADII_M, key=lambda x: abs(x - args.radius)))
            per_trip_rows.append({"partition": part, "trip_id": tid, "fixes": len(fx), "valid_fixes": int(fx.fix_valid.sum()),
                                  "misses": int((fx.fix_valid & ~fx.true_in_cands).sum()),
                                  "fixes_without_candidates": int((fx.n_cand == 0).sum()),
                                  "mean_candidates": float(fx.n_cand.mean()), "candidate_rows": len(cand),
                                  "nearest_top1_hits": chosen["nearest_top1_hits"]})
            print(f"[{part} {k:>2}/{len(trips)}] {tid}: {len(fx):>6,} fixes, {int(fx.fix_valid.sum()):>6,} labelled, "
                  f"{len(cand):>8,} candidate rows ({time.time() - ts:.1f}s)", flush=True)
        for name, frames in ((f"candidates_{part}.parquet", cands), (f"fixes_{part}.parquet", fixes)):
            df = pd.concat(frames, ignore_index=True)
            path = proc_dir / name
            df.to_parquet(path, index=False)
            sizes[name] = (len(df), path.stat().st_size / 1e6)

    radius_df = pd.DataFrame(radius_rows)
    agg = aggregate_radius(radius_df)
    agg.to_csv(out_dir / "miss_rate.csv", index=False)
    per_trip = pd.DataFrame(per_trip_rows)
    per_trip["miss_rate"] = per_trip["misses"] / per_trip["valid_fixes"].where(per_trip["valid_fixes"] > 0)
    per_trip.to_csv(out_dir / "miss_rate_per_trip.csv", index=False)
    label_stats = pd.DataFrame(label_rows)
    label_stats.to_csv(out_dir / "label_stats.csv", index=False)

    figs = make_figures(agg, label_stats, args.radius, fig_dir)
    write_summary(agg, per_trip, label_stats, args.radius, sizes, figs, out_dir, time.time() - t0)
    print(agg.to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s -> {proc_dir}, {out_dir}")


if __name__ == "__main__":
    main()
