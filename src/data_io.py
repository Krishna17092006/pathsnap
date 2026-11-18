"""Raw-file loading for the PathSnap dataset.

Layout:  data/<partition>/<trip_id>/<trip_id>.{track,nodes,arcs,route}

All four files are headerless and whitespace separated (the .track files use
CRLF line endings, which whitespace splitting absorbs):

    .track  lon lat t   GPS observations, t in seconds, nominally 1 Hz
    .nodes  lon lat     road-network nodes; 0-based row index = node_id
    .arcs   u v         directed road segments u -> v; 0-based row index = arc_id
    .route  arc_id      reference route as chronological arc ids (may be absent)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PARTITIONS = ("train", "student_test")
FILE_TYPES = ("track", "nodes", "arcs", "route")
COLUMNS = {
    "track": ["lon", "lat", "t"],
    "nodes": ["lon", "lat"],
    "arcs": ["u", "v"],
    "route": ["arc_id"],
}


def list_trips(partition: str, data_dir: Path = DATA_DIR) -> list[str]:
    part_dir = Path(data_dir) / partition
    if not part_dir.is_dir():
        return []
    return sorted(p.name for p in part_dir.iterdir() if p.is_dir())


def trip_file(trip_id: str, partition: str, kind: str, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / partition / trip_id / f"{trip_id}.{kind}"


def read_raw(path: Path, kind: str) -> pd.DataFrame:
    """Read one raw file as-is (no cleaning).

    Returns an empty frame with the expected columns if the file is missing or
    empty. Raises ValueError if the column count differs from the format spec.
    Short rows become NaN, so missing values stay visible to the caller.
    """
    cols = COLUMNS[kind]
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    if df.shape[1] != len(cols):
        raise ValueError(f"{path.name}: expected {len(cols)} columns, found {df.shape[1]}")
    df.columns = cols
    return df


def load_raw(trip_id: str, partition: str, data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    return {k: read_raw(trip_file(trip_id, partition, k, data_dir), k) for k in FILE_TYPES}


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame        # lon, lat, t  (float64; invalid tokens -> NaN)
    nodes: pd.DataFrame        # lon, lat     (float64)
    arcs: pd.DataFrame         # u, v         (int64)
    route: np.ndarray | None   # arc ids (int64), or None when no reference route is supplied

    @property
    def has_route(self) -> bool:
        return self.route is not None and len(self.route) > 0


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").astype("float64")


def trip_from_raw(trip_id: str, partition: str, raw: dict[str, pd.DataFrame]) -> Trip:
    arcs = _numeric(raw["arcs"])
    if arcs.isna().any().any():
        raise ValueError(f"{trip_id}: non-numeric or missing values in .arcs")
    route_col = _numeric(raw["route"])["arc_id"]
    if route_col.isna().any():
        raise ValueError(f"{trip_id}: non-numeric or missing values in .route")
    return Trip(
        trip_id=trip_id,
        partition=partition,
        track=_numeric(raw["track"]),
        nodes=_numeric(raw["nodes"]),
        arcs=arcs.astype("int64"),
        route=route_col.to_numpy(dtype="int64") if len(route_col) else None,
    )


def load_trip(trip_id: str, partition: str, data_dir: Path = DATA_DIR) -> Trip:
    return trip_from_raw(trip_id, partition, load_raw(trip_id, partition, data_dir))


def iter_trips(partitions=PARTITIONS, data_dir: Path = DATA_DIR):
    for part in partitions:
        for tid in list_trips(part, data_dir):
            yield load_trip(tid, part, data_dir)
