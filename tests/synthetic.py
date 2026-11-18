"""Synthetic trips in the raw dataset format, for tests only (never for reported results).

Each trip is a grid of streets (mostly two-way, some one-way) with a vehicle driving a
random route at ~12 m/s, sampled at 1 Hz with Gaussian GPS noise. Labels and candidate
tables are then built with the project's real labels.py / candidates.py code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.candidates import build_tables
from src.data_io import load_trip
from src.geo import LocalProjection
from src.labels import make_labels
from src.network import RoadNetwork

GRID = 7
SPACING_M = 80.0
SPEED_MS = 12.0
NOISE_M = 4.0


def _grid(rng):
    ids = np.arange(GRID * GRID).reshape(GRID, GRID)
    edges = []
    for r in range(GRID):
        for c in range(GRID):
            for dr, dc in ((0, 1), (1, 0)):
                if r + dr < GRID and c + dc < GRID:
                    a, b = ids[r, c], ids[r + dr, c + dc]
                    if rng.random() < 0.15:
                        edges.append((a, b) if rng.random() < 0.5 else (b, a))   # one-way
                    else:
                        edges += [(a, b), (b, a)]
    xy = np.array([(c * SPACING_M, r * SPACING_M) for r in range(GRID) for c in range(GRID)], dtype=float)
    return xy, np.array(edges, dtype=int)


def _route(arcs, rng, n_arcs=18):
    out_of = {}
    for i, (u, v) in enumerate(arcs):
        out_of.setdefault(u, []).append(i)
    cur = int(rng.choice(arcs[:, 0]))
    route, prev_u = [], -1
    for _ in range(n_arcs):
        opts = [a for a in out_of.get(cur, []) if arcs[a, 1] != prev_u] or out_of.get(cur, [])
        if not opts:
            break
        a = int(rng.choice(opts))
        route.append(a)
        prev_u, cur = cur, int(arcs[a, 1])
    return np.array(route, dtype=int)


def _drive(xy, arcs, route, rng, gap_at=None):
    pts = []
    carry = 0.0
    for a in route:
        p, q = xy[arcs[a, 0]], xy[arcs[a, 1]]
        length = np.linalg.norm(q - p)
        s = carry
        while s < length:
            pts.append(p + (q - p) * (s / length))
            s += SPEED_MS
        carry = s - length
    pts = np.array(pts) + rng.normal(0, NOISE_M, (len(pts), 2))
    t = np.arange(len(pts), dtype=float)
    if gap_at is not None and gap_at < len(t):
        t[gap_at:] += 40.0
    return pts, t


def write_trip(data_dir: Path, partition: str, trip_id: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    xy, arcs = _grid(rng)
    route = _route(arcs, rng)
    pts, t = _drive(xy, arcs, route, rng, gap_at=20 if seed % 2 else None)
    proj = LocalProjection(10.0 + seed * 0.5, 50.0 - seed * 0.3)
    nlon, nlat = proj.to_lonlat(xy[:, 0], xy[:, 1])
    tlon, tlat = proj.to_lonlat(pts[:, 0], pts[:, 1])
    d = Path(data_dir) / partition / trip_id
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / f"{trip_id}.track", np.column_stack([tlon, tlat, t]), fmt="%.8f")
    np.savetxt(d / f"{trip_id}.nodes", np.column_stack([nlon, nlat]), fmt="%.8f")
    np.savetxt(d / f"{trip_id}.arcs", arcs, fmt="%d")
    np.savetxt(d / f"{trip_id}.route", route, fmt="%d")


def build_processed(data_dir: Path, processed_dir: Path, partition: str, trip_ids: list[str], radius_m=50.0):
    """Same steps as src.build_dataset.process_trip, but for an arbitrary data_dir."""
    cands, fixes = [], []
    for tid in trip_ids:
        trip = load_trip(tid, partition, data_dir)
        proj = LocalProjection.from_points(trip.track["lon"], trip.track["lat"])
        net = RoadNetwork(trip.nodes, trip.arcs, proj)
        px, py = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
        labels = make_labels(px, py, net, trip.route)
        c, f = build_tables(trip, net, px, py, labels, radius_m)
        cands.append(c)
        fixes.append(f)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    pd.concat(cands, ignore_index=True).to_parquet(Path(processed_dir) / f"candidates_{partition}.parquet", index=False)
    pd.concat(fixes, ignore_index=True).to_parquet(Path(processed_dir) / f"fixes_{partition}.parquet", index=False)


def make_dataset(root: Path, n_train: int = 6, n_test: int = 2):
    data_dir, processed_dir = Path(root) / "data", Path(root) / "data" / "processed"
    parts = {"train": [f"{i:08d}" for i in range(1, n_train + 1)],
             "student_test": [f"{i:08d}" for i in range(100, 100 + n_test)]}
    for part, ids in parts.items():
        for tid in ids:
            write_trip(data_dir, part, tid, seed=int(tid))
        build_processed(data_dir, processed_dir, part, ids)
    return data_dir, processed_dir
