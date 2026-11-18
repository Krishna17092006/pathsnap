"""Candidate generation: every road arc within a radius of each GPS fix.

Produces the two shared tables described in TEAM.md section 2.2:

* candidate table: one row per (fix, candidate arc)
* fix table:       one row per GPS fix (label, coverage, number of candidates)

A candidate's `label` is 1 when it is the fix's matched route arc *or an exact
duplicate of it* (same start and end node). Some networks contain duplicate
arcs with identical geometry; no model could tell them apart, so treating them
as negatives would only add label noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import FixLabels
from .network import RoadNetwork

CANDIDATE_COLUMNS = ["trip_id", "partition", "fix_idx", "t", "lon", "lat", "arc_id", "dist_m", "t_along",
                     "cand_rank", "n_cand", "true_arc", "fix_valid", "label"]
FIX_COLUMNS = ["trip_id", "partition", "fix_idx", "t", "lon", "lat", "route_pos", "true_arc", "match_dist_m",
               "fix_valid", "n_cand", "true_in_cands"]


def candidate_pairs(net: RoadNetwork, px, py, radius_m: float, labels: FixLabels) -> pd.DataFrame:
    """All (fix, arc) pairs within radius_m, ranked by distance, with labels attached."""
    cand = net.arcs_within(px, py, radius_m).rename(columns={"point_idx": "fix_idx", "t": "t_along"})
    cand = cand.sort_values(["fix_idx", "dist_m", "arc_id"], kind="stable").reset_index(drop=True)
    fix = cand["fix_idx"].to_numpy()
    arc = cand["arc_id"].to_numpy()
    cand["cand_rank"] = cand.groupby("fix_idx").cumcount().astype("int32")
    cand["n_cand"] = np.bincount(fix, minlength=len(px))[fix].astype("int32")
    true_arc = labels.true_arc[fix]
    has = true_arc >= 0
    safe = np.where(has, true_arc, 0)
    same_geom = (net.u[arc] == net.u[safe]) & (net.v[arc] == net.v[safe])
    cand["true_arc"] = true_arc
    cand["fix_valid"] = labels.fix_valid[fix]
    cand["label"] = (has & same_geom).astype("int8")
    return cand


def build_tables(trip, net: RoadNetwork, px, py, labels: FixLabels, radius_m: float,
                 fix_idx_map=None, pairs: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate and fix tables for one trip.

    fix_idx_map maps positions in px/py back to row indices of the raw .track file
    (identity when every fix is valid). `pairs` may be a candidate_pairs() result
    for a radius >= radius_m, to avoid a second spatial query.
    """
    n = len(px)
    fix_idx_map = np.arange(n) if fix_idx_map is None else np.asarray(fix_idx_map)
    tr = trip.track.iloc[fix_idx_map]
    lon, lat, t = tr["lon"].to_numpy(), tr["lat"].to_numpy(), tr["t"].to_numpy()

    if pairs is None:
        cand = candidate_pairs(net, px, py, radius_m, labels)
    else:
        cand = pairs[pairs["dist_m"] <= radius_m].reset_index(drop=True)
        f = cand["fix_idx"].to_numpy()
        cand["cand_rank"] = cand.groupby("fix_idx").cumcount().astype("int32")
        cand["n_cand"] = np.bincount(f, minlength=n)[f].astype("int32")
    pos = cand["fix_idx"].to_numpy()
    cand["t"] = t[pos]
    cand["lon"] = lon[pos]
    cand["lat"] = lat[pos]
    cand["fix_idx"] = fix_idx_map[pos]
    cand["true_arc"] = cand["true_arc"].astype("int64")
    cand["trip_id"] = trip.trip_id
    cand["partition"] = trip.partition
    cand = cand[CANDIDATE_COLUMNS]

    n_cand = np.bincount(pos, minlength=n)
    true_in = np.zeros(n, dtype=bool)
    true_in[pos[cand["label"].to_numpy() == 1]] = True
    fixes = pd.DataFrame({
        "trip_id": trip.trip_id, "partition": trip.partition, "fix_idx": fix_idx_map, "t": t, "lon": lon, "lat": lat,
        "route_pos": labels.route_pos, "true_arc": labels.true_arc, "match_dist_m": labels.match_dist_m,
        "fix_valid": labels.fix_valid, "n_cand": n_cand.astype("int32"), "true_in_cands": true_in & labels.fix_valid,
    })[FIX_COLUMNS]
    return cand, fixes
