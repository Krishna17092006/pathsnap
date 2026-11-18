"""Feature engineering for (GPS fix, candidate arc) rows.

Every feature is geometric or kinematic and computed only from the GPS track
and the road network. Nothing here reads `.route`, `true_arc` or `label`, so
the same code runs unchanged at inference time on trips without a reference.
Arc ids and trip ids are never features: every trip has its own network.

Run from the project root (needs data/processed/candidates_*.parquet):
    python -m src.features            # writes data/processed/features_<partition>.parquet

Feature groups and why they help
    distance     dist_m, dist_minus_min, dist_ratio, cand_rank, n_cand
                 GPS error is small (median ~5 m), so closer arcs are likelier; the
                 distance *relative to the nearest candidate* matters more than the
                 absolute value, because the whole candidate set shifts with GPS bias.
    direction    heading_diff_deg, cos_heading_diff, has_heading, cos_heading_weighted
                 A two-way road is two arcs at the same distance; only the direction of
                 travel separates them (nearest-candidate Top-1 is ~54% for this reason).
                 GPS heading is reliable above ~8 m/s and jitter below ~2.5 m/s, so
                 cos_heading_weighted scales the agreement by a speed-based confidence.
    kinematics   speed_mps, log_dt_prev, log_dt_next
                 Speed tells the model how far to trust heading; gaps mark fixes where
                 neighbouring information is stale.
    arc shape    log_arc_length, t_along, is_two_way, out_degree_v, in_degree_u
                 Projections that land on an arc end (t_along 0 or 1) are ambiguous at
                 junctions; topology describes where the arc sits in the network.
    context      prev_nearest_link, next_nearest_link
                 Whether the arc continues from / into the nearest arc of the previous /
                 next fix. Uses only geometry of neighbouring fixes, no labels.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR
from .data_io import DATA_DIR, load_trip
from .geo import LocalProjection
from .network import RoadNetwork

FEATURE_COLUMNS = [
    "dist_m", "dist_minus_min", "dist_ratio", "cand_rank", "n_cand",
    "heading_diff_deg", "cos_heading_diff", "has_heading", "cos_heading_weighted",
    "speed_mps", "log_dt_prev", "log_dt_next",
    "log_arc_length", "t_along", "is_two_way", "out_degree_v", "in_degree_u",
    "prev_nearest_link", "next_nearest_link",
]
# Non-feature columns added for evaluation (route continuity needs arc endpoints).
META_COLUMNS = ["arc_u", "arc_v"]

REQUIRED_INPUT = ["trip_id", "fix_idx", "arc_id", "dist_m", "t_along", "cand_rank", "n_cand"]
FORBIDDEN_FEATURES = {"arc_id", "trip_id", "true_arc", "label", "fix_valid", "route_pos"}

MIN_HEADING_MOVE_M = 3.0      # displacement below this over i-1 -> i+1 gives no usable heading
MAX_HEADING_DT_S = 30.0       # heading across a longer gap is meaningless
RELIABLE_SPEED_MS = 8.0       # from the label direction check: heading is trustworthy above this
MAX_SPEED_MS = 70.0           # faster than 252 km/h is a position jump, not driving
MIN_ARC_LEN_M = 0.5           # shorter arcs have no defined direction

assert not FORBIDDEN_FEATURES & set(FEATURE_COLUMNS), "label-derived or id columns must never be features"


def fix_kinematics(track: pd.DataFrame, proj: LocalProjection) -> pd.DataFrame:
    """Per-fix heading (deg, clockwise from north, NaN if unknown), speed and time gaps.

    Heading and speed use the central difference i-1 -> i+1, which halves GPS jitter
    compared with a one-step difference. End points fall back to one-sided steps.
    """
    n = len(track)
    x, y = proj.to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    t = track["t"].to_numpy(dtype="float64")
    i = np.arange(n)
    lo, hi = np.maximum(i - 1, 0), np.minimum(i + 1, n - 1)
    dx, dy, dt = x[hi] - x[lo], y[hi] - y[lo], t[hi] - t[lo]
    move = np.hypot(dx, dy)

    heading = np.degrees(np.arctan2(dx, dy)) % 360.0
    heading[(move < MIN_HEADING_MOVE_M) | (dt > MAX_HEADING_DT_S) | (dt <= 0) | ~np.isfinite(move)] = np.nan

    speed = np.where(dt > 0, move / np.where(dt > 0, dt, 1.0), 0.0)
    speed = np.clip(np.nan_to_num(speed, nan=0.0), 0.0, MAX_SPEED_MS)

    gaps = np.diff(t) if n > 1 else np.array([])
    dt_prev = np.concatenate([[np.nan], gaps])
    dt_next = np.concatenate([gaps, [np.nan]])
    dt_prev = np.where(np.isnan(dt_prev), dt_next, dt_prev)   # first fix: borrow the next gap
    dt_next = np.where(np.isnan(dt_next), dt_prev, dt_next)   # last fix: borrow the previous gap
    dt_prev = np.nan_to_num(dt_prev, nan=1.0)
    dt_next = np.nan_to_num(dt_next, nan=1.0)
    return pd.DataFrame({
        "fix_idx": i, "heading_deg": heading, "speed_mps": speed,
        "log_dt_prev": np.log1p(np.clip(dt_prev, 0, None)), "log_dt_next": np.log1p(np.clip(dt_next, 0, None)),
    })


def _arc_tables(net: RoadNetwork) -> dict[str, np.ndarray]:
    """Per-arc bearing, length and topology, indexed by arc id."""
    bearing = np.degrees(np.arctan2(net.bx - net.ax, net.by - net.ay)) % 360.0
    seg = np.hypot(net.bx - net.ax, net.by - net.ay)
    bearing[seg < MIN_ARC_LEN_M] = np.nan
    key = net.u * net.n_nodes + net.v
    reverse = np.isin(net.v * net.n_nodes + net.u, key) & (net.u != net.v)
    out_deg = np.bincount(net.u, minlength=net.n_nodes)
    in_deg = np.bincount(net.v, minlength=net.n_nodes)
    return {"bearing": bearing, "length": net.length_m, "two_way": reverse.astype("int8"),
            "out_v": out_deg[net.v], "in_u": in_deg[net.u]}


def _context_links(fix_idx: np.ndarray, arc: np.ndarray, rank: np.ndarray, net: RoadNetwork):
    """Does each candidate continue from the previous fix's nearest arc / into the next one's?"""
    nearest = pd.Series(arc[rank == 0], index=fix_idx[rank == 0])
    prev_near = nearest.reindex(fix_idx - 1).to_numpy()
    next_near = nearest.reindex(fix_idx + 1).to_numpy()
    has_p, has_n = ~np.isnan(prev_near), ~np.isnan(next_near)
    pn = np.where(has_p, prev_near, 0).astype("int64")
    nn = np.where(has_n, next_near, 0).astype("int64")
    prev_link = has_p & ((arc == pn) | (net.u[arc] == net.v[pn]))
    next_link = has_n & ((arc == nn) | (net.v[arc] == net.u[nn]))
    return prev_link.astype("int8"), next_link.astype("int8")


def add_features(cand_df: pd.DataFrame, trip, net: RoadNetwork) -> pd.DataFrame:
    """Add FEATURE_COLUMNS and META_COLUMNS to one trip's candidate rows; keeps all input columns.

    `cand_df` must contain REQUIRED_INPUT for a single trip. `fix_idx` indexes rows of
    `trip.track`. Returns a new frame in the input row order.
    """
    missing = [c for c in REQUIRED_INPUT if c not in cand_df.columns]
    if missing:
        raise ValueError(f"candidate table is missing columns: {missing}")
    if cand_df["trip_id"].nunique() > 1:
        raise ValueError("add_features expects the rows of a single trip")
    out = cand_df.copy()
    if out.empty:
        for c in FEATURE_COLUMNS + META_COLUMNS:
            out[c] = pd.Series(dtype="float64")
        return out

    fix = out["fix_idx"].to_numpy(dtype="int64")
    arc = out["arc_id"].to_numpy(dtype="int64")
    if fix.min() < 0 or fix.max() >= len(trip.track):
        raise ValueError("fix_idx outside the trip's track")
    if arc.min() < 0 or arc.max() >= net.n_arcs:
        raise ValueError("arc_id outside the trip's network")

    # ---- distance
    dist = out["dist_m"].to_numpy(dtype="float64")
    dmin = out.groupby("fix_idx")["dist_m"].transform("min").to_numpy()
    out["dist_minus_min"] = dist - dmin
    out["dist_ratio"] = (dist + 1.0) / (dmin + 1.0)

    # ---- direction and kinematics
    proj = LocalProjection.from_points(trip.track["lon"], trip.track["lat"])
    kin = fix_kinematics(trip.track, proj)
    arcs = _arc_tables(net)
    heading = kin["heading_deg"].to_numpy()[fix]
    bearing = arcs["bearing"][arc]
    diff = np.abs((heading - bearing + 180.0) % 360.0 - 180.0)
    known = np.isfinite(diff)
    speed = kin["speed_mps"].to_numpy()[fix]
    cos_diff = np.where(known, np.cos(np.radians(np.where(known, diff, 0.0))), 0.0)
    out["heading_diff_deg"] = np.where(known, diff, 90.0)   # unknown = uninformative (perpendicular)
    out["cos_heading_diff"] = cos_diff
    out["has_heading"] = known.astype("int8")
    out["cos_heading_weighted"] = cos_diff * np.clip(speed / RELIABLE_SPEED_MS, 0.0, 1.0)
    out["speed_mps"] = speed
    out["log_dt_prev"] = kin["log_dt_prev"].to_numpy()[fix]
    out["log_dt_next"] = kin["log_dt_next"].to_numpy()[fix]

    # ---- arc shape and topology
    out["log_arc_length"] = np.log1p(arcs["length"][arc])
    out["is_two_way"] = arcs["two_way"][arc]
    out["out_degree_v"] = arcs["out_v"][arc]
    out["in_degree_u"] = arcs["in_u"][arc]

    # ---- context from neighbouring fixes
    out["prev_nearest_link"], out["next_nearest_link"] = _context_links(
        fix, arc, out["cand_rank"].to_numpy(), net)

    out["arc_u"] = net.u[arc]
    out["arc_v"] = net.v[arc]
    return out


def build_features(partition: str, processed_dir: Path = PROCESSED_DIR, data_dir: Path = DATA_DIR,
                   trips: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Features for every trip in data/processed/candidates_<partition>.parquet."""
    processed_dir = Path(processed_dir)
    cand = pd.read_parquet(processed_dir / f"candidates_{partition}.parquet")
    if trips is not None:
        cand = cand[cand["trip_id"].isin(trips)]
    parts = []
    for k, (tid, rows) in enumerate(cand.groupby("trip_id", sort=True), 1):
        ts = time.time()
        trip = load_trip(tid, partition, data_dir)
        proj = LocalProjection.from_points(trip.track["lon"], trip.track["lat"])
        net = RoadNetwork(trip.nodes, trip.arcs, proj)
        parts.append(add_features(rows, trip, net))
        if verbose:
            print(f"[features {partition}] {k:>2} {tid}: {len(rows):>8,} rows ({time.time() - ts:.1f}s)", flush=True)
    if not parts:
        return add_features(cand.iloc[:0], None, None)
    return pd.concat(parts, ignore_index=True)


def features_path(partition: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    return Path(processed_dir) / f"features_{partition}.parquet"


def load_or_build(partition: str, processed_dir: Path = PROCESSED_DIR, data_dir: Path = DATA_DIR,
                  rebuild: bool = False) -> pd.DataFrame:
    """Cached feature table; rebuilt when missing, stale (older than the candidates) or on request."""
    path = features_path(partition, processed_dir)
    src = Path(processed_dir) / f"candidates_{partition}.parquet"
    stale = path.exists() and src.exists() and path.stat().st_mtime < src.stat().st_mtime
    if path.exists() and not rebuild and not stale:
        df = pd.read_parquet(path)
        if set(FEATURE_COLUMNS + META_COLUMNS) <= set(df.columns):
            return df
    df = build_features(partition, processed_dir, data_dir)
    df.to_parquet(path, index=False)
    return df


def save_feature_list(path: Path) -> Path:
    """Ordered model-input columns, e.g. for artifacts/features.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build candidate features")
    ap.add_argument("--partition", choices=["train", "student_test", "all"], default="all")
    args = ap.parse_args(argv)
    for part in (["train", "student_test"] if args.partition == "all" else [args.partition]):
        t0 = time.time()
        df = load_or_build(part, rebuild=True)
        print(f"{part}: {len(df):,} rows, {len(FEATURE_COLUMNS)} features -> {features_path(part)} "
              f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
