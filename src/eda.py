"""Exploratory data analysis for PathSnap.

Only the train and student_test partitions are analysed. Evaluator data is never read.

Run from the project root:
    python -m src.eda            # full run
    python -m src.eda --quick    # 3 trips per partition, for a fast smoke test

Outputs
    results/eda/*.csv            per-trip tables, partition summary, data-quality log
    results/eda/figures/*.png    static figures for the report
    results/eda/eda_summary.md   factual summary of the measured numbers
    maps/eda/*.html              interactive Folium maps
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from . import plot_style as ps
from .data_io import COLUMNS, FILE_TYPES, PARTITIONS, PROJECT_ROOT, list_trips, load_raw, load_trip, trip_file, trip_from_raw
from .geo import LocalProjection, haversine_m
from .maps import overview_map, trip_map
from .network import RoadNetwork

RESULTS_DIR = PROJECT_ROOT / "results" / "eda"
FIG_DIR = RESULTS_DIR / "figures"
MAP_DIR = PROJECT_ROOT / "maps" / "eda"

SEED = 42
SPEED_OUTLIER_MS = 70.0          # 252 km/h between consecutive fixes: a position jump, not driving
STATIONARY_STEP_M = 0.5          # consecutive fixes closer than this count as "not moving"
ROUTE_DIST_THRESHOLDS_M = (10, 25, 50, 100)
CANDIDATE_RADII_M = (25, 50, 100)
MAX_CANDIDATE_SAMPLE = 2000      # GPS points per trip used for candidate-density statistics
DT_BINS = [(-np.inf, 0, "<= 0 s"), (0, 1, "1 s"), (1, 2, "2 s"), (2, 5, "3-5 s"),
           (5, 30, "6-30 s"), (30, 300, "31-300 s"), (300, np.inf, "> 300 s")]


# =====================================================================================
# Per-trip analysis
# =====================================================================================
def analyse_trip(trip_id: str, partition: str, rng: np.random.Generator) -> dict:
    key = {"partition": partition, "trip_id": trip_id}
    raw = load_raw(trip_id, partition)

    # ---------------- inventory + missing values (on the raw, uncleaned files)
    inventory = dict(key)
    missing = []
    for kind in FILE_TYPES:
        f = trip_file(trip_id, partition, kind)
        inventory[f"{kind}_present"] = f.exists()
        inventory[f"{kind}_kb"] = round(f.stat().st_size / 1024, 1) if f.exists() else 0.0
        inventory[f"{kind}_rows"] = len(raw[kind])
        for col in COLUMNS[kind]:
            num = pd.to_numeric(raw[kind][col], errors="coerce")
            missing.append({**key, "file": kind, "column": col, "rows": len(num),
                            "raw_dtype": str(raw[kind][col].dtype),
                            "missing_or_non_numeric": int(num.isna().sum())})
    trip = trip_from_raw(trip_id, partition, raw)

    # ---------------- GPS track quality
    lon, lat, t = (trip.track[c].to_numpy() for c in ("lon", "lat", "t"))
    invalid = (~np.isfinite(lon) | ~np.isfinite(lat) | ~np.isfinite(t)
               | (np.abs(lat) > 90) | (np.abs(lon) > 180) | ((lon == 0) & (lat == 0)))
    lon, lat, t = lon[~invalid], lat[~invalid], t[~invalid]
    dt = np.diff(t)
    step_m = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    moving = dt > 0
    speed = step_m[moving] / dt[moving]
    lat_c, lon_c = float(np.mean(lat)), float(np.mean(lon))
    gps = {
        **key,
        "n_points": int(len(trip.track)),
        "n_invalid_points": int(invalid.sum()),
        "t_first_s": float(t[0]),
        "duration_min": float((t[-1] - t[0]) / 60),
        "n_fractional_timestamps": int(np.sum(t % 1 != 0)),
        "n_nonincreasing_t": int(np.sum(dt <= 0)),
        "n_duplicate_t": int(np.sum(dt == 0)),
        "share_dt_1s": float(np.mean(dt == 1)),
        "n_gaps_gt5s": int(np.sum(dt > 5)),
        "n_gaps_gt30s": int(np.sum(dt > 30)),
        "n_gaps_gt300s": int(np.sum(dt > 300)),
        "max_gap_s": float(dt.max()) if len(dt) else 0.0,
        "n_stationary_steps": int(np.sum(step_m < STATIONARY_STEP_M)),
        "median_speed_kmh": float(np.median(speed) * 3.6),
        "p95_speed_kmh": float(np.percentile(speed, 95) * 3.6),
        "max_speed_kmh": float(speed.max() * 3.6),
        "n_speed_outliers": int(np.sum(speed > SPEED_OUTLIER_MS)),
        "path_length_km": float(step_m.sum() / 1000),
        "lon_min": float(lon.min()), "lon_max": float(lon.max()),
        "lat_min": float(lat.min()), "lat_max": float(lat.max()),
        "centroid_lon": lon_c, "centroid_lat": lat_c,
        "extent_ew_km": float(haversine_m(lon.min(), lat_c, lon.max(), lat_c) / 1000),
        "extent_ns_km": float(haversine_m(lon_c, lat.min(), lon_c, lat.max()) / 1000),
    }

    # ---------------- projection check: local equirectangular vs haversine
    proj = LocalProjection.from_points(lon, lat)
    px, py = proj.to_xy(lon, lat)
    nlon, nlat = trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy()
    nx, ny = proj.to_xy(nlon, nlat)
    i = rng.integers(0, len(lon), 3000)
    j = rng.integers(0, len(nlon), 3000)
    d_true = haversine_m(lon[i], lat[i], nlon[j], nlat[j])
    d_proj = np.hypot(px[i] - nx[j], py[i] - ny[j])
    far = d_true > 100
    gps["proj_max_rel_err_pct"] = float(np.max(np.abs(d_proj - d_true)[far] / d_true[far]) * 100) if far.any() else 0.0
    gps["proj_max_abs_err_m"] = float(np.max(np.abs(d_proj - d_true)))

    # ---------------- road network
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    n, u, v = net.n_nodes, net.u, net.v
    code, rev = u * n + v, v * n + u
    L = net.length_m
    network = {
        **key,
        "n_nodes": n, "n_arcs": net.n_arcs,
        "self_loops": int(np.sum(u == v)),
        "duplicate_arcs": int(net.n_arcs - len(np.unique(code))),
        "two_way_share": float(np.mean(np.isin(rev, code))),
        "arc_len_min_m": float(L.min()), "arc_len_median_m": float(np.median(L)),
        "arc_len_mean_m": float(L.mean()), "arc_len_p95_m": float(np.percentile(L, 95)),
        "arc_len_max_m": float(L.max()),
        "zero_length_arcs": int(np.sum(L < 0.01)),
        "directed_road_km": float(L.sum() / 1000),
        **net.connectivity_stats(),
    }
    # how far the supplied network extends beyond the GPS bounding box (negative = track leaves the network)
    margins = [haversine_m(lon.min(), lat_c, nlon.min(), lat_c) * np.sign(lon.min() - nlon.min()),
               haversine_m(nlon.max(), lat_c, lon.max(), lat_c) * np.sign(nlon.max() - lon.max()),
               haversine_m(lon_c, lat.min(), lon_c, nlat.min()) * np.sign(lat.min() - nlat.min()),
               haversine_m(lon_c, nlat.max(), lon_c, lat.max()) * np.sign(nlat.max() - lat.max())]
    network["min_bbox_margin_km"] = float(min(margins) / 1000)

    # ---------------- reference route
    route = trip.route if trip.has_route else np.array([], dtype="int64")
    bad_ids = (route < 0) | (route >= net.n_arcs)
    r = route[~bad_ids]
    in_scc = net.largest_scc_mask[u[r]] & net.largest_scc_mask[v[r]]
    route_len_km = float(L[r].sum() / 1000)
    route_stats = {
        **key,
        "n_route_arcs": int(len(route)),
        "invalid_arc_ids": int(bad_ids.sum()),
        "unique_arcs": int(len(np.unique(r))),
        "revisited_arcs": int(len(r) - len(np.unique(r))),
        "consecutive_duplicates": int(np.sum(r[1:] == r[:-1])),
        "breaks": int(len(net.route_breaks(r))),
        "u_turns": int(np.sum((u[r[1:]] == v[r[:-1]]) & (v[r[1:]] == u[r[:-1]]))),
        "route_length_km": route_len_km,
        "gps_to_route_length_ratio": float(gps["path_length_km"] / route_len_km) if route_len_km else np.nan,
        "route_arcs_outside_largest_scc": int((~in_scc).sum()),
    }

    # ---------------- GPS <-> reference route alignment
    alignment = dict(key)
    arrays = {"dt": dt, "speed_kmh": speed * 3.6, "arc_len": L}
    if len(r):
        d_route, near_arc = net.distance_to_arcs(px, py, np.unique(r))
        first_pos = {}
        for pos, a in enumerate(r):
            first_pos.setdefault(int(a), pos)
        order = np.array([first_pos[int(a)] for a in near_arc])
        d_first, _ = net.distance_to_arcs(px[:1], py[:1], r[:1])
        d_last, _ = net.distance_to_arcs(px[-1:], py[-1:], r[-1:])
        alignment.update({
            "dist_route_median_m": float(np.median(d_route)),
            "dist_route_p90_m": float(np.percentile(d_route, 90)),
            "dist_route_p95_m": float(np.percentile(d_route, 95)),
            "dist_route_p99_m": float(np.percentile(d_route, 99)),
            "dist_route_max_m": float(d_route.max()),
            **{f"share_within_{thr}m_of_route": float(np.mean(d_route <= thr)) for thr in ROUTE_DIST_THRESHOLDS_M},
            "start_dist_to_first_route_arc_m": float(d_first[0]),
            "end_dist_to_last_route_arc_m": float(d_last[0]),
            # if the nearest route arc jumps backwards along the route, naive nearest-arc labelling is ambiguous
            "nearest_route_arc_backward_share": float(np.mean(np.diff(order) < 0)) if len(order) > 1 else 0.0,
        })
        arrays["dist_route"] = d_route

    # ---------------- candidate density (how many arcs sit near each GPS point)
    idx = np.unique(np.linspace(0, len(px) - 1, min(MAX_CANDIDATE_SAMPLE, len(px))).astype(int))
    cand = net.arcs_within(px[idx], py[idx], max(CANDIDATE_RADII_M))
    for rad in CANDIDATE_RADII_M:
        counts = np.bincount(cand.loc[cand["dist_m"] <= rad, "point_idx"], minlength=len(idx))
        arrays[f"cand_{rad}"] = counts
        alignment[f"cand_median_within_{rad}m"] = float(np.median(counts))
        alignment[f"cand_p95_within_{rad}m"] = float(np.percentile(counts, 95))
        alignment[f"share_no_arc_within_{rad}m"] = float(np.mean(counts == 0))
    if len(r) and len(cand):
        on_route = np.zeros(net.n_arcs, dtype=bool)
        on_route[r] = True
        cand["on_route"] = on_route[cand["arc_id"].to_numpy()]
        nearest_d = cand.groupby("point_idx")["dist_m"].transform("min")
        tied = cand[cand["dist_m"] <= nearest_d + 1e-6]
        hit = tied.groupby("point_idx")["on_route"].any().reindex(range(len(idx)), fill_value=False)
        alignment["nearest_arc_on_route_share"] = float(hit.mean())

    return {"inventory": inventory, "missing": missing, "gps": gps, "network": network,
            "route": route_stats, "alignment": alignment, "arrays": arrays}


# =====================================================================================
# Figures
# =====================================================================================
def _concat(per_trip_arrays, partition_of, name, part):
    parts = [a[name] for tid, a in per_trip_arrays.items() if partition_of[tid] == part and name in a]
    return np.concatenate(parts) if parts else np.array([])


def _strip(ax, df, col, xlabel, rng, scale="linear", pct=False):
    parts = [p for p in PARTITIONS if (df["partition"] == p).any()]
    for row, part in enumerate(parts):
        vals = df.loc[df["partition"] == part, col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)] * (100 if pct else 1)
        if not len(vals):
            continue
        y = row + rng.uniform(-0.17, 0.17, len(vals))
        ax.scatter(vals, y, s=28, color=ps.PARTITION_COLOR[part], edgecolors=ps.SURFACE, linewidths=0.8, zorder=3)
        med = np.median(vals)
        ax.plot([med, med], [row - 0.33, row + 0.33], color=ps.INK, lw=1.8, zorder=4, solid_capstyle="round")
    ax.set_yticks(range(len(parts)),
                  [f"{ps.PARTITION_LABEL[p]}\n{int((df['partition'] == p).sum())} trips" for p in parts])
    ax.set_ylim(len(parts) - 0.45, -0.55)
    if scale == "symlog":
        ax.set_xscale("symlog", linthresh=1)
    elif scale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def _note(fig, text):
    fig.text(0.01, -0.02, text, fontsize=8, color=ps.MUTED, ha="left", va="top")


def make_figures(tables: dict, arrays: dict, partition_of: dict, examples: list[tuple[str, str]], rng) -> list[Path]:
    ps.apply_style()
    gps, net, rt, al = tables["gps"], tables["network"], tables["route"], tables["alignment"]
    parts = [p for p in PARTITIONS if (gps["partition"] == p).any()]
    out = []

    # 1. trip sizes
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9))
    _strip(axes[0], gps, "n_points", "GPS observations per trip (log)", rng, "log")
    _strip(axes[1], gps, "duration_min", "Trip duration, minutes (log)", rng, "log")
    _strip(axes[2], gps, "path_length_km", "GPS path length, km (log)", rng, "log")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Trip size by partition", x=0.01, ha="left", fontweight="bold", fontsize=11)
    _note(fig, "Each dot is one trip; the black tick marks the partition median.")
    out.append(ps.save(fig, FIG_DIR / "fig01_trip_sizes.png"))

    # 2. sampling intervals and gaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    labels = [b[2] for b in DT_BINS]
    shares = {}
    for part in parts:
        d = _concat(arrays, partition_of, "dt", part)
        shares[part] = np.array([np.mean((d > lo) & (d <= hi)) * 100 for lo, hi, _ in DT_BINS])
    keep = [k for k in range(len(labels)) if any(shares[p][k] > 0 for p in parts)]
    h = 0.38
    for pi, part in enumerate(parts):
        ys = np.arange(len(keep)) + (pi - (len(parts) - 1) / 2) * (h + 0.04)
        vals = shares[part][keep]
        ax.barh(ys, np.where(vals > 0, vals, np.nan), height=h, color=ps.PARTITION_COLOR[part],
                label=ps.PARTITION_LABEL[part], zorder=3)
        for y, val in zip(ys, vals):
            if val > 0:
                ax.text(val * 1.15, y, f"{val:.3g}%", va="center", fontsize=7.5, color=ps.INK_2)
    ax.set_yticks(np.arange(len(keep)), [labels[k] for k in keep])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(right=400)
    ax.set_xlabel("Share of consecutive-fix intervals, % (log)")
    ax.set_title("Sampling interval between consecutive GPS fixes")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _strip(axes[1], gps, "max_gap_s", "Largest gap in the trip, seconds (log)", rng, "log")
    axes[1].set_title("Largest time gap per trip")
    out.append(ps.save(fig, FIG_DIR / "fig02_sampling_and_gaps.png"))

    # 3. speed distribution
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    bins = np.arange(0, 202, 2)
    for part in parts:
        s = _concat(arrays, partition_of, "speed_kmh", part)
        ax.hist(np.clip(s, 0, 200), bins=bins, density=True, histtype="step", lw=2,
                color=ps.PARTITION_COLOR[part], label=f"{ps.PARTITION_LABEL[part]} (n={len(s):,})")
    n_out = int(gps["n_speed_outliers"].sum())
    ax.set_xlabel("Speed between consecutive fixes, km/h (values above 200 shown at 200)")
    ax.set_ylabel("Density")
    ax.set_title("Implied speed distribution")
    ax.legend()
    _note(fig, f"{n_out} intervals imply more than {SPEED_OUTLIER_MS * 3.6:.0f} km/h (position jumps).")
    out.append(ps.save(fig, FIG_DIR / "fig03_speed_distribution.png"))

    # 4. GPS-to-route distance
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for part in parts:
        d = np.sort(_concat(arrays, partition_of, "dist_route", part))
        if not len(d):
            continue
        y = np.arange(1, len(d) + 1) / len(d) * 100
        ax.plot(np.maximum(d, 0.05), y, color=ps.PARTITION_COLOR[part], lw=2, label=ps.PARTITION_LABEL[part])
    for thr in (25, 50):
        ax.axvline(thr, color=ps.BASELINE, lw=1, ls="--", zorder=1)
        ax.text(thr * 1.05, 4, f"{thr} m", color=ps.MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 2000)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Distance from GPS fix to nearest reference-route arc, m (log)")
    ax.set_ylabel("Cumulative % of GPS fixes")
    ax.set_title("How far GPS fixes lie from the reference route")
    ax.legend(loc="upper left")
    _strip(axes[1], al, "nearest_arc_on_route_share", "% of fixes whose nearest road arc is on the route", rng, pct=True)
    axes[1].set_title("Naive nearest-road baseline, per trip")
    out.append(ps.save(fig, FIG_DIR / "fig04_gps_to_route_distance.png"))

    # 5. candidate density
    fig, axes = plt.subplots(1, len(CANDIDATE_RADII_M), figsize=(12, 3.0), sharey=False)
    for ax, rad in zip(axes, CANDIDATE_RADII_M):
        allc = np.concatenate([_concat(arrays, partition_of, f"cand_{rad}", p) for p in parts])
        hi = int(np.percentile(allc, 99)) + 2
        bins = np.arange(0, hi + 1) - 0.5
        for part in parts:
            c = _concat(arrays, partition_of, f"cand_{rad}", part)
            ax.hist(np.clip(c, 0, hi - 1), bins=bins, density=True, histtype="step", lw=2,
                    color=ps.PARTITION_COLOR[part], label=f"{ps.PARTITION_LABEL[part]} (median {np.median(c):.0f})")
        ax.set_title(f"Arcs within {rad} m of a GPS fix")
        ax.set_xlabel("Number of candidate arcs")
        ax.legend()
    axes[0].set_ylabel("Density")
    _note(fig, f"Up to {MAX_CANDIDATE_SAMPLE} evenly spaced fixes per trip; two-way roads count as two arcs.")
    out.append(ps.save(fig, FIG_DIR / "fig05_candidate_density.png"))

    # 6. network size and arc length
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ax = axes[0]
    for part in parts:
        d = net[net["partition"] == part]
        ax.scatter(d["n_nodes"], d["n_arcs"], s=30, color=ps.PARTITION_COLOR[part], edgecolors=ps.SURFACE,
                   linewidths=0.8, label=ps.PARTITION_LABEL[part], zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nodes (log)")
    ax.set_ylabel("Arcs (log)")
    ax.set_title("Road-network size per trip")
    ax.legend()
    ax = axes[1]
    bins = np.logspace(-1, 4, 80)
    for part in parts:
        a = _concat(arrays, partition_of, "arc_len", part)
        ax.hist(np.clip(a, 0.1, 1e4), bins=bins, density=False, histtype="step", lw=2,
                weights=np.full(len(a), 100 / len(a)), color=ps.PARTITION_COLOR[part],
                label=f"{ps.PARTITION_LABEL[part]} (median {np.median(a):.0f} m)")
    ax.set_xscale("log")
    ax.set_xlabel("Arc length, m (log)")
    ax.set_ylabel("% of arcs")
    ax.set_title("Road-segment length")
    ax.legend()
    out.append(ps.save(fig, FIG_DIR / "fig06_network_size.png"))

    # 7. connectivity
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9))
    _strip(axes[0], net, "weak_components", "Weakly connected components (symlog)", rng, "symlog")
    _strip(axes[1], net, "largest_scc_share", "% of nodes in largest strongly connected component", rng, pct=True)
    _strip(axes[2], net, "isolated_nodes", "Isolated nodes (symlog)", rng, "symlog")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Road-network connectivity", x=0.01, ha="left", fontweight="bold", fontsize=11)
    _note(fig, "Each dot is one trip; the black tick marks the partition median.")
    out.append(ps.save(fig, FIG_DIR / "fig07_connectivity.png"))

    # 8. reference-route quality
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9))
    _strip(axes[0], rt, "gps_to_route_length_ratio", "GPS path length / route length", rng)
    _strip(axes[1], rt, "breaks", "Disconnected consecutive route arcs (symlog)", rng, "symlog")
    _strip(axes[2], al, "nearest_route_arc_backward_share", "% of steps where nearest route arc moves backwards",
           rng, pct=True)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Reference-route consistency", x=0.01, ha="left", fontweight="bold", fontsize=11)
    _note(fig, "Each dot is one trip; the black tick marks the partition median.")
    out.append(ps.save(fig, FIG_DIR / "fig08_route_quality.png"))

    # 9. trajectory examples
    if examples:
        ncol = 3
        nrow = int(np.ceil(len(examples) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, (part, tid) in zip(axes, examples):
            _plot_trajectory(ax, part, tid)
        for ax in axes[len(examples):]:
            ax.set_visible(False)
        handles = [plt.Line2D([], [], color=ps.NETWORK_COLOR, lw=1.2, label="Road network (within 250 m)"),
                   plt.Line2D([], [], color=ps.ROUTE_COLOR, lw=3, label="Reference route"),
                   plt.Line2D([], [], color=ps.GPS_COLOR, marker="o", ls="", ms=4, label="GPS observations")]
        fig.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.01, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out.append(ps.save(fig, FIG_DIR / "fig09_trajectory_examples.png"))

        # 10. zoomed windows: GPS noise relative to the road geometry
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4.2 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, (part, tid) in zip(axes, examples):
            _plot_trajectory(ax, part, tid, zoom_half_m=300)
        for ax in axes[len(examples):]:
            ax.set_visible(False)
        handles[0] = plt.Line2D([], [], color=ps.NETWORK_COLOR, lw=1.2, label="Road network")
        fig.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.01, 1.02))
        _note(fig, "600 m windows centred on the fix whose distance to the route equals the trip's 95th percentile.")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out.append(ps.save(fig, FIG_DIR / "fig10_zoomed_gps_noise.png"))
    return out


def _segments(net, arc_ids, ox=0.0, oy=0.0, scale=1.0):
    return np.stack([np.column_stack([net.ax[arc_ids] - ox, net.ay[arc_ids] - oy]),
                     np.column_stack([net.bx[arc_ids] - ox, net.by[arc_ids] - oy])], axis=1) / scale


def _plot_trajectory(ax, part, tid, zoom_half_m: float | None = None):
    """Whole trip in km, or (zoom_half_m set) a window around the fix whose route distance is the trip's p95."""
    from matplotlib.ticker import MaxNLocator

    trip = load_trip(tid, part)
    tr = trip.track.dropna()
    lon, lat = tr["lon"].to_numpy(), tr["lat"].to_numpy()
    proj = LocalProjection.from_points(lon, lat)
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    px, py = proj.to_xy(lon, lat)
    title = f"{ps.PARTITION_LABEL[part]} {tid}  ·  {len(tr):,} fixes"

    if zoom_half_m is None:
        ox = oy = 0.0
        scale, unit = 1000.0, "km"
        stride = max(1, len(px) // 1500)
        near = np.unique(net.arcs_within(px[::stride], py[::stride], 250)["arc_id"])
        gps_size, net_lw = 1.5, 0.6
    else:
        d, _ = net.distance_to_arcs(px, py, np.unique(trip.route))
        c = int(np.argmin(np.abs(d - np.percentile(d, 95))))
        ox, oy = px[c], py[c]
        scale, unit = 1.0, "m"
        near = np.unique(net.arcs_within(px[c:c + 1], py[c:c + 1], zoom_half_m * 1.5)["arc_id"])
        gps_size, net_lw = 16, 1.2
        title = f"{ps.PARTITION_LABEL[part]} {tid}  ·  fix #{c}, {d[c]:.0f} m from route"

    ax.add_collection(LineCollection(_segments(net, near, ox, oy, scale), colors=ps.NETWORK_COLOR,
                                     linewidths=net_lw, zorder=1))
    if trip.has_route:
        ax.add_collection(LineCollection(_segments(net, trip.route, ox, oy, scale), colors=ps.ROUTE_COLOR,
                                         linewidths=2.6 if zoom_half_m is None else 4, zorder=2, capstyle="round"))
    gx, gy = (px - ox) / scale, (py - oy) / scale
    if zoom_half_m is not None:
        ax.plot(gx, gy, color=ps.GPS_COLOR, lw=0.8, alpha=0.6, zorder=3)
    ax.scatter(gx, gy, s=gps_size, color=ps.GPS_COLOR, zorder=4,
               edgecolors=ps.SURFACE if zoom_half_m else "none", linewidths=0.8 if zoom_half_m else 0)
    ax.set_aspect("equal")
    if zoom_half_m is None:
        ax.autoscale_view()
    else:
        ax.set_xlim(-zoom_half_m, zoom_half_m)
        ax.set_ylim(-zoom_half_m, zoom_half_m)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel(f"East, {unit}")
    ax.set_ylabel(f"North, {unit}")
    ax.grid(False)


# =====================================================================================
# Summary writers
# =====================================================================================
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def quality_issues(t: dict) -> pd.DataFrame:
    rows = []

    def add(df, mask, issue, col):
        for _, r in df[mask].iterrows():
            rows.append({"partition": r["partition"], "trip_id": r["trip_id"], "issue": issue, "value": r[col]})

    g, n, rt, al = t["gps"], t["network"], t["route"], t["alignment"]
    add(g, g["n_invalid_points"] > 0, "invalid GPS coordinates", "n_invalid_points")
    add(g, g["n_nonincreasing_t"] > 0, "non-increasing timestamps", "n_nonincreasing_t")
    add(g, g["max_gap_s"] > 300, "GPS gap longer than 300 s (max gap, s)", "max_gap_s")
    add(g, g["n_speed_outliers"] > 0, f"intervals faster than {SPEED_OUTLIER_MS * 3.6:.0f} km/h", "n_speed_outliers")
    add(n, n["self_loops"] > 0, "self-loop arcs", "self_loops")
    add(n, n["duplicate_arcs"] > 0, "duplicate arcs", "duplicate_arcs")
    add(n, n["zero_length_arcs"] > 0, "zero-length arcs", "zero_length_arcs")
    add(n, n["min_bbox_margin_km"] < 0, "GPS track extends beyond network bounding box (km)", "min_bbox_margin_km")
    add(rt, rt["invalid_arc_ids"] > 0, "route arc id outside arc table", "invalid_arc_ids")
    add(rt, rt["breaks"] > 0, "route has disconnected consecutive arcs", "breaks")
    add(rt, rt["u_turns"] > 0, "route contains U-turns", "u_turns")
    add(rt, rt["route_arcs_outside_largest_scc"] > 0, "route arcs outside largest strongly connected component",
        "route_arcs_outside_largest_scc")
    if "dist_route_p95_m" in al:
        add(al, al["dist_route_p95_m"] > 50, "95th percentile GPS-to-route distance above 50 m", "dist_route_p95_m")
        add(al, al["start_dist_to_first_route_arc_m"] > 50, "first GPS fix > 50 m from first route arc",
            "start_dist_to_first_route_arc_m")
        add(al, al["end_dist_to_last_route_arc_m"] > 50, "last GPS fix > 50 m from last route arc",
            "end_dist_to_last_route_arc_m")
    return pd.DataFrame(rows, columns=["partition", "trip_id", "issue", "value"])


def partition_summary(t: dict) -> pd.DataFrame:
    metrics = [("gps", "n_points"), ("gps", "duration_min"), ("gps", "path_length_km"), ("gps", "share_dt_1s"),
               ("gps", "max_gap_s"), ("gps", "median_speed_kmh"), ("gps", "proj_max_rel_err_pct"),
               ("network", "n_nodes"), ("network", "n_arcs"), ("network", "arc_len_median_m"),
               ("network", "two_way_share"), ("network", "isolated_nodes"), ("network", "weak_components"),
               ("network", "largest_scc_share"), ("route", "n_route_arcs"), ("route", "route_length_km"),
               ("route", "gps_to_route_length_ratio"), ("route", "breaks"),
               ("alignment", "dist_route_median_m"), ("alignment", "dist_route_p95_m"),
               ("alignment", "share_within_25m_of_route"), ("alignment", "share_within_50m_of_route"),
               ("alignment", "cand_median_within_25m"), ("alignment", "cand_median_within_50m"),
               ("alignment", "nearest_arc_on_route_share"), ("alignment", "nearest_route_arc_backward_share")]
    rows = []
    for table, col in metrics:
        df = t[table]
        if col not in df:
            continue
        for part, grp in df.groupby("partition"):
            s = grp[col].astype(float)
            rows.append({"metric": col, "partition": part, "trips": len(s), "sum": s.sum(), "min": s.min(),
                         "median": s.median(), "mean": s.mean(), "max": s.max()})
    return pd.DataFrame(rows)


def write_summary(t: dict, issues: pd.DataFrame, arrays: dict, partition_of: dict, figs, maps, elapsed: float) -> Path:
    g, n, rt, al, miss = t["gps"], t["network"], t["route"], t["alignment"], t["missing"]
    parts = [p for p in PARTITIONS if (g["partition"] == p).any()]
    f = lambda x, d=1: f"{x:,.{d}f}"  # noqa: E731
    L = ["# PathSnap EDA summary", "",
         "Generated by `python -m src.eda`. Every number below is computed from the raw files in `data/`.",
         "Only the `train` and `student_test` partitions are analysed; evaluator trips are not used.", ""]

    inv = []
    for p in parts:
        gp, np_, rp = g[g.partition == p], n[n.partition == p], rt[rt.partition == p]
        inv.append({"Partition": ps.PARTITION_LABEL[p], "Trips": len(gp), "GPS fixes (total)": f"{gp.n_points.sum():,}",
                    "Fixes/trip min-median-max": f"{gp.n_points.min():,} / {int(gp.n_points.median()):,} / {gp.n_points.max():,}",
                    "Nodes/trip (median)": f"{int(np_.n_nodes.median()):,}", "Arcs/trip (median)": f"{int(np_.n_arcs.median()):,}",
                    "Route arcs/trip (median)": f"{int(rp.n_route_arcs.median()):,}"})
    L += ["## 1. Inventory", "", _md_table(pd.DataFrame(inv)), "",
          f"Every trip folder contains all four files: {bool(t['inventory'][[c for c in t['inventory'] if c.endswith('_present')]].all().all())}.", ""]

    L += ["## 2. Missing and invalid values", "",
          f"- Missing or non-numeric cells across all files and columns: **{int(miss.missing_or_non_numeric.sum())}**",
          f"- Invalid GPS fixes (NaN, out-of-range, or exactly 0,0): **{int(g.n_invalid_points.sum())}**",
          f"- Non-increasing timestamps: **{int(g.n_nonincreasing_t.sum())}**; fractional timestamps: **{int(g.n_fractional_timestamps.sum())}**",
          f"- Raw dtypes per column are listed in `missing_values.csv`.", ""]

    L += ["## 3. Sampling and gaps", ""]
    for p in parts:
        d = _concat(arrays, partition_of, "dt", p)
        gp = g[g.partition == p]
        worst = gp.loc[gp.max_gap_s.idxmax()]
        L.append(f"- {ps.PARTITION_LABEL[p]}: {np.mean(d == 1) * 100:.2f}% of intervals are exactly 1 s; "
                 f"{int(np.sum(d > 5)):,} gaps > 5 s, {int(np.sum(d > 30)):,} > 30 s, {int(np.sum(d > 300)):,} > 300 s. "
                 f"Largest gap: {f(worst.max_gap_s, 0)} s (trip {worst.trip_id}).")
    L.append(f"- Trips with a gap longer than 300 s: {', '.join(g.loc[g.max_gap_s > 300, 'trip_id']) or 'none'}")
    L += ["", "## 4. Speeds and position jumps", ""]
    for p in parts:
        gp = g[g.partition == p]
        L.append(f"- {ps.PARTITION_LABEL[p]}: median of per-trip median speed {f(gp.median_speed_kmh.median())} km/h; "
                 f"{int(gp.n_speed_outliers.sum())} intervals faster than {SPEED_OUTLIER_MS * 3.6:.0f} km/h "
                 f"in {int((gp.n_speed_outliers > 0).sum())} trips; {int(gp.n_stationary_steps.sum()):,} stationary steps (< {STATIONARY_STEP_M} m).")
    L += ["", "## 5. Coordinate system check", "",
          f"- Coordinate order is (longitude, latitude): latitudes lie in [{f(g.lat_min.min(), 2)}, {f(g.lat_max.max(), 2)}] "
          f"and the GPS fixes fall onto the supplied road network (section 8).",
          f"- Largest trip extent: {f(g.extent_ew_km.max())} km east-west, {f(g.extent_ns_km.max())} km north-south.",
          f"- Local equirectangular projection vs haversine (random fix-node pairs > 100 m apart, up to tens of km): "
          f"max relative error {g.proj_max_rel_err_pct.max():.3f}% over all trips (median trip "
          f"{g.proj_max_rel_err_pct.median():.3f}%). Distortion is a scale error, so at candidate-search distances "
          f"(<= 100 m) the error is below {g.proj_max_rel_err_pct.max():.3f} m; the projection is adequate for geometry, "
          f"and haversine is used for reported lengths.", ""]

    L += ["## 6. Road networks", ""]
    for p in parts:
        np_ = n[n.partition == p]
        L.append(f"- {ps.PARTITION_LABEL[p]}: two-way share {np_.two_way_share.min() * 100:.0f}-{np_.two_way_share.max() * 100:.0f}%; "
                 f"median arc length {f(np_.arc_len_median_m.median())} m (max {f(np_.arc_len_max_m.max(), 0)} m); "
                 f"isolated nodes total {int(np_.isolated_nodes.sum()):,}; weakly connected components per trip "
                 f"{int(np_.weak_components.min())}-{int(np_.weak_components.max())}; largest strongly connected component "
                 f"holds {np_.largest_scc_share.min() * 100:.1f}-{np_.largest_scc_share.max() * 100:.1f}% of nodes; "
                 f"self-loops {int(np_.self_loops.sum())}, duplicate arcs {int(np_.duplicate_arcs.sum())}, zero-length arcs {int(np_.zero_length_arcs.sum())}.")
    L += ["", "## 7. Reference routes", ""]
    for p in parts:
        rp = rt[rt.partition == p]
        L.append(f"- {ps.PARTITION_LABEL[p]}: {int(rp.breaks.sum())} breaks in {int((rp.breaks > 0).sum())} trips; "
                 f"{int(rp.u_turns.sum())} U-turns; {int(rp.revisited_arcs.sum())} revisited arcs; "
                 f"{int(rp.invalid_arc_ids.sum())} invalid arc ids; {int(rp.route_arcs_outside_largest_scc.sum())} route arcs outside "
                 f"the largest strongly connected component; GPS/route length ratio {f(rp.gps_to_route_length_ratio.min(), 2)}-"
                 f"{f(rp.gps_to_route_length_ratio.max(), 2)} (median {f(rp.gps_to_route_length_ratio.median(), 2)}).")
    L += ["", "## 8. GPS to reference-route alignment", ""]
    for p in parts:
        d = _concat(arrays, partition_of, "dist_route", p)
        ap = al[al.partition == p]
        if not len(d):
            continue
        L.append(f"- {ps.PARTITION_LABEL[p]}: distance to route median {f(np.median(d))} m, p90 {f(np.percentile(d, 90))} m, "
                 f"p95 {f(np.percentile(d, 95))} m, p99 {f(np.percentile(d, 99))} m; within 10/25/50/100 m: "
                 + " / ".join(f"{np.mean(d <= thr) * 100:.1f}%" for thr in ROUTE_DIST_THRESHOLDS_M) + ".")
        L.append(f"  - Naive nearest-arc baseline (nearest arc lies on the route): median over trips "
                 f"{ap.nearest_arc_on_route_share.median() * 100:.1f}% (range {ap.nearest_arc_on_route_share.min() * 100:.1f}-"
                 f"{ap.nearest_arc_on_route_share.max() * 100:.1f}%).")
        L.append(f"  - Nearest route arc moves backwards along the route in {ap.nearest_route_arc_backward_share.median() * 100:.1f}% "
                 f"of steps (median over trips).")
        gp = g[g.partition == p].merge(ap, on=["partition", "trip_id"])
        far = ((1 - gp.share_within_100m_of_route) * gp.n_points).round().astype(int)
        if far.sum() > 0:
            top = gp.assign(far=far).sort_values("far", ascending=False).head(3)
            parts_txt = ", ".join(f"{r.trip_id} ({r.far:,} of {r.n_points:,} fixes)" for r in top.itertuples() if r.far > 0)
            L.append(f"  - Fixes more than 100 m from the route: {int(far.sum()):,}; largest contributors: {parts_txt}.")
    L += ["", "## 9. Candidate density", ""]
    for p in parts:
        ap = al[al.partition == p]
        L.append(f"- {ps.PARTITION_LABEL[p]}: median arcs within 25/50/100 m = "
                 + " / ".join(f"{np.median(_concat(arrays, partition_of, f'cand_{rad}', p)):.0f}" for rad in CANDIDATE_RADII_M)
                 + "; fixes with no arc within 25/50 m: "
                 + " / ".join(f"{np.mean(_concat(arrays, partition_of, f'cand_{rad}', p) == 0) * 100:.2f}%" for rad in (25, 50)) + ".")
    L += ["", "## 10. Data-quality log", "",
          f"{len(issues)} flagged items across {issues.trip_id.nunique() if len(issues) else 0} trips (full list in `data_quality_issues.csv`):", ""]
    if len(issues):
        L.append(_md_table(issues.groupby("issue").agg(trips=("trip_id", "nunique")).reset_index()))
    L += ["", "## 11. Output files", "", "Figures:"] + [f"- `{p.relative_to(PROJECT_ROOT).as_posix()}`" for p in figs]
    L += ["", "Maps:"] + [f"- `{p.relative_to(PROJECT_ROOT).as_posix()}`" for p in maps]
    L += ["", f"Runtime: {elapsed:.0f} s.", ""]
    path = RESULTS_DIR / "eda_summary.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# =====================================================================================
# Main
# =====================================================================================
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="PathSnap exploratory data analysis")
    ap.add_argument("--quick", action="store_true", help="analyse only 3 trips per partition")
    ap.add_argument("--no-maps", action="store_true", help="skip Folium map generation")
    args = ap.parse_args(argv)

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    collected = {k: [] for k in ("inventory", "missing", "gps", "network", "route", "alignment")}
    arrays, partition_of = {}, {}
    for part in PARTITIONS:
        trips = list_trips(part)
        if args.quick:
            trips = trips[:3]
        for k, tid in enumerate(trips, 1):
            ts = time.time()
            res = analyse_trip(tid, part, rng)
            for name in collected:
                (collected[name].extend if name == "missing" else collected[name].append)(res[name])
            arrays[tid], partition_of[tid] = res["arrays"], part
            print(f"[{part} {k:>2}/{len(trips)}] {tid}: {res['gps']['n_points']:>6,} fixes, "
                  f"{res['network']['n_arcs']:>7,} arcs  ({time.time() - ts:.1f}s)", flush=True)

    tables = {k: pd.DataFrame(v) for k, v in collected.items()}
    for name, df in tables.items():
        df.to_csv(RESULTS_DIR / f"{'missing_values' if name == 'missing' else name + '_stats'}.csv", index=False)
    issues = quality_issues(tables)
    issues.to_csv(RESULTS_DIR / "data_quality_issues.csv", index=False)
    partition_summary(tables).to_csv(RESULTS_DIR / "partition_summary.csv", index=False)

    # Trips chosen for figures and maps: spread of sizes in student_test plus notable train trips.
    g, al, rt = tables["gps"], tables["alignment"], tables["route"]
    st = g[g.partition == "student_test"].sort_values("n_points")
    tr = g[g.partition == "train"]
    examples = []
    if len(st):
        examples += [("student_test", st.iloc[i].trip_id) for i in sorted({0, len(st) // 2, len(st) - 1})]
    if len(tr):
        typical = tr.iloc[(tr.n_points - tr.n_points.median()).abs().argmin()].trip_id
        largest_gap = tr.loc[tr.max_gap_s.idxmax()].trip_id
        tr_al = al[al.partition == "train"]
        worst_aligned = (tr_al.loc[tr_al.share_within_50m_of_route.idxmin()].trip_id
                         if "share_within_50m_of_route" in tr_al else typical)
        for tid in (typical, largest_gap, worst_aligned):
            if ("train", tid) not in examples:
                examples.append(("train", tid))
    pd.DataFrame(examples, columns=["partition", "trip_id"]).to_csv(RESULTS_DIR / "selected_trips.csv", index=False)

    figs = make_figures(tables, arrays, partition_of, examples, np.random.default_rng(SEED))
    print(f"figures: {len(figs)} written to {FIG_DIR.relative_to(PROJECT_ROOT)}")

    maps = []
    if not args.no_maps:
        inv = tables["inventory"].merge(g, on=["partition", "trip_id"]).merge(
            tables["network"][["partition", "trip_id", "n_nodes", "n_arcs"]], on=["partition", "trip_id"])
        p = MAP_DIR / "overview_trip_locations.html"
        overview_map(inv).save(str(p))
        maps.append(p)
        map_trips = [("student_test", tid) for tid in st.trip_id] + [e for e in examples if e[0] == "train"]
        for part, tid in map_trips:
            trip = load_trip(tid, part)
            tr_ok = trip.track.dropna()
            lon, lat = tr_ok["lon"].to_numpy(), tr_ok["lat"].to_numpy()
            proj = LocalProjection.from_points(lon, lat)
            net = RoadNetwork(trip.nodes, trip.arcs, proj)
            d = None
            if trip.has_route and len(tr_ok) == len(trip.track):
                px, py = proj.to_xy(lon, lat)
                d, _ = net.distance_to_arcs(px, py, np.unique(trip.route))
            p = MAP_DIR / f"{part}_{tid}.html"
            trip_map(trip, net, dist_to_route=d).save(str(p))
            maps.append(p)
        print(f"maps: {len(maps)} written to {MAP_DIR.relative_to(PROJECT_ROOT)}")

    summary = write_summary(tables, issues, arrays, partition_of, figs, maps, time.time() - t0)
    print(f"summary: {summary.relative_to(PROJECT_ROOT)}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
