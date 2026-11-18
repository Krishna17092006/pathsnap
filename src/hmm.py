"""Hidden Markov Model and Viterbi decoder for PathSnap map matching.

States are candidate directed road arcs at each GPS fix. Emissions are either
classifier probabilities (preferred) or a distance-based Gaussian fallback.
Transitions compare network route distance with observed haversine displacement.
All dynamic programming is performed in log space.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import OrderedDict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from .geo import haversine_m


@dataclass(frozen=True)
class DecoderConfig:
    beta: float = 50.0                 # transition mismatch scale, metres
    sigma: float = 8.0                 # distance-emission scale, metres
    gap_threshold_s: float = 60.0      # split a trip after this GPS time gap
    emission_weight: float = 1.0       # multiplier on log emission score
    max_speed_mps: float = 50.0        # plausible route-distance cap
    min_transition_m: float = 200.0    # lower bound on dijkstra search limit
    probability_epsilon: float = 1e-6

    def __post_init__(self):
        if self.beta <= 0 or self.sigma <= 0 or self.gap_threshold_s <= 0:
            raise ValueError("beta, sigma and gap_threshold_s must be positive")
        if self.emission_weight < 0 or self.max_speed_mps <= 0 or self.min_transition_m <= 0:
            raise ValueError("emission_weight must be non-negative; speed and distance limits positive")

    def to_dict(self) -> dict:
        return asdict(self)


class _TransitionCache:
    """Per-trip directed shortest-path helper; caches Dijkstra rows by source node and limit."""
    def __init__(self, net):
        self.net = net
        # Collapse duplicate directed arcs to the shortest edge weight; COO/CSR
        # would otherwise sum duplicate entries and overstate route distance.
        edge_frame = pd.DataFrame({"u": net.u, "v": net.v, "w": net.length_m})
        edge_frame = edge_frame.groupby(["u", "v"], as_index=False)["w"].min()
        self.graph = coo_matrix(
            (edge_frame["w"].to_numpy(dtype="float64"),
             (edge_frame["u"].to_numpy(dtype="int64"), edge_frame["v"].to_numpy(dtype="int64"))),
            shape=(net.n_nodes, net.n_nodes),
        ).tocsr()
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self.max_cached_rows = 128

    def row(self, source: int, bucket: int) -> np.ndarray:
        """Get one bounded, LRU-cached directed Dijkstra distance row."""
        key = (int(source), int(bucket))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        values = dijkstra(
            self.graph, directed=True, indices=int(source), limit=float(bucket),
            return_predecessors=False,
        )
        self._cache[key] = values
        if len(self._cache) > self.max_cached_rows:
            self._cache.popitem(last=False)
        return values

    def distance(self, source: int, target: int, limit_m: float) -> float:
        if source == target:
            return 0.0
        # Round limit upward to reusable buckets; exact feasibility is checked
        # by the transition scorer after route-distance assembly.
        bucket = int(np.ceil(limit_m / 50.0) * 50)
        return float(self.row(source, bucket)[int(target)])


def _emission_scores(group: pd.DataFrame, probs: np.ndarray | None,
                     sigma: float, eps: float) -> np.ndarray:
    if probs is None:
        d = np.maximum(group["dist_m"].to_numpy(dtype="float64"), 0.0)
        return -0.5 * np.square(d / sigma)
    p = np.asarray(probs, dtype="float64")
    p = np.nan_to_num(p, nan=eps, posinf=1.0, neginf=eps)
    p = np.clip(p, eps, 1.0)
    return np.log(p)


def _transition_scores(prev: pd.DataFrame, curr: pd.DataFrame, net,
                       gps_distance_m: float, dt_s: float,
                       cfg: DecoderConfig, cache: _TransitionCache) -> np.ndarray:
    """Return [previous candidate, current candidate] log transition scores.

    Route-distance assembly is vectorised over candidate pairs. Dijkstra is
    still run only from the unique end nodes of the previous candidate arcs.
    """
    a = prev["arc_id"].to_numpy(dtype="int64")
    b = curr["arc_id"].to_numpy(dtype="int64")
    ta = np.clip(prev["t_along"].to_numpy(dtype="float64"), 0.0, 1.0)
    tb = np.clip(curr["t_along"].to_numpy(dtype="float64"), 0.0, 1.0)
    la = net.length_m[a]
    lb = net.length_m[b]
    limit = max(cfg.min_transition_m, cfg.max_speed_mps * max(float(dt_s), 0.0))

    source_nodes = np.unique(net.v[a])
    target_nodes = np.unique(net.u[b])
    bucket = int(np.ceil(limit / 50.0) * 50)
    rows = [cache.row(int(source), bucket)[target_nodes] for source in source_nodes]
    shortest = np.asarray(rows, dtype="float64")
    source_pos = np.searchsorted(source_nodes, net.v[a])
    target_pos = np.searchsorted(target_nodes, net.u[b])
    middle = shortest[source_pos[:, None], target_pos[None, :]]

    remaining = (1.0 - ta) * la
    route = remaining[:, None] + middle + (tb * lb)[None, :]

    # Exact duplicate directed arcs are the same physical segment. Moving
    # forward on that segment is shorter than leaving and re-entering it.
    same_segment = ((net.u[a][:, None] == net.u[b][None, :])
                    & (net.v[a][:, None] == net.v[b][None, :]))
    forward = same_segment & (tb[None, :] >= ta[:, None] - 1e-6)
    direct = np.maximum(0.0, (tb[None, :] - ta[:, None]) * la[:, None])
    route = np.where(forward, direct, route)

    feasible = np.isfinite(route) & (route <= limit)
    mismatch = np.abs(route - max(0.0, float(gps_distance_m)))
    score = -mismatch / cfg.beta
    score[~feasible] = -np.inf
    return score


def _traceback_block(back: list[np.ndarray], dp: np.ndarray, start: int, end: int,
                     state_counts: list[int]) -> list[int]:
    """Trace one connected Viterbi block, inclusive state-time indices."""
    path = [0] * (end - start + 1)
    state = int(np.argmax(dp))
    path[-1] = state
    for local_t in range(end - start, 0, -1):
        bp = back[local_t]
        state = int(bp[state])
        if state < 0:
            state = 0
        path[local_t - 1] = state
    return path


def viterbi_decode(candidates: pd.DataFrame, fixes: pd.DataFrame, net,
                   probabilities: Iterable[float] | None = None,
                   config: DecoderConfig | None = None) -> np.ndarray:
    """Decode one trip.

    Parameters
    ----------
    candidates : DataFrame
        One row per (fix, directed arc), requiring fix_idx, arc_id, dist_m,
        t_along. Rows may optionally contain `prob` but explicit `probabilities`
        takes precedence. Probabilities must align with the input row order.
    fixes : DataFrame
        One row per raw GPS fix, requiring fix_idx, t, lon, lat. Rows with no
        candidates are left at -1 (unmatched); transitions bridge across them.
    net : RoadNetwork
        Network belonging to this trip.

    Returns
    -------
    np.ndarray
        Predicted arc id per row of `fixes`; -1 means no candidate/unmatched.
    """
    cfg = config or DecoderConfig()
    output = np.full(len(fixes), -1, dtype="int64")
    if candidates.empty or fixes.empty:
        return output
    required_c = {"fix_idx", "arc_id", "dist_m", "t_along"}
    required_f = {"fix_idx", "t", "lon", "lat"}
    if not required_c.issubset(candidates.columns):
        raise ValueError(f"candidates missing columns: {sorted(required_c - set(candidates.columns))}")
    if not required_f.issubset(fixes.columns):
        raise ValueError(f"fixes missing columns: {sorted(required_f - set(fixes.columns))}")

    cand = candidates.copy().reset_index(drop=True)
    probs = None if probabilities is None else np.asarray(list(probabilities), dtype="float64")
    if probs is not None and len(probs) != len(cand):
        raise ValueError("probabilities must have one value per candidate row")
    if probs is None and "prob" in cand.columns:
        probs = cand["prob"].to_numpy(dtype="float64")

    fix_pos = {int(fid): i for i, fid in enumerate(fixes["fix_idx"].to_numpy())}
    valid_cand = cand["fix_idx"].isin(fix_pos)
    cand = cand.loc[valid_cand].copy()
    if probs is not None:
        probs = probs[valid_cand.to_numpy()]
    if cand.empty:
        return output

    # Group candidate states by fix, preserving stable candidate ordering.
    cand["_input_order"] = np.arange(len(cand))
    groups: list[tuple[int, pd.DataFrame, np.ndarray]] = []
    for fix_id, g in cand.groupby("fix_idx", sort=True):
        row_idx = g["_input_order"].to_numpy(dtype="int64")
        p = probs[row_idx] if probs is not None else None
        groups.append((int(fix_id), g.drop(columns="_input_order").reset_index(drop=True),
                       _emission_scores(g, p, cfg.sigma, cfg.probability_epsilon)))

    # Viterbi runs over candidate-bearing fixes. Missing-candidate fixes remain -1;
    # the next transition spans the elapsed time and GPS displacement since the
    # previous candidate-bearing fix, effectively carrying the previous state.
    cache = _TransitionCache(net)
    block_items: list[tuple[int, pd.DataFrame, np.ndarray]] = []
    back: list[np.ndarray] = []
    dp: np.ndarray | None = None

    def close_block():
        """Trace the current independent temporal segment and release its memory."""
        nonlocal block_items, back, dp
        if block_items and dp is not None:
            state = int(np.argmax(dp))
            states = [0] * len(block_items)
            states[-1] = state
            for local_t in range(len(block_items) - 1, 0, -1):
                state = int(back[local_t][state])
                if state < 0:
                    state = 0
                states[local_t - 1] = state
            for item, state_idx in zip(block_items, states):
                fix_id, group, _ = item
                output[fix_pos[fix_id]] = int(group.iloc[state_idx]["arc_id"])
        block_items, back, dp = [], [], None

    for item in groups:
        fix_id, g, emission = item
        current_fix = fixes.iloc[fix_pos[fix_id]]
        if not block_items:
            block_items = [item]
            dp = cfg.emission_weight * emission
            back = [np.full(len(g), -1, dtype="int64")]
            continue

        prev_id, prev_g, _ = block_items[-1]
        prev_fix = fixes.iloc[fix_pos[prev_id]]
        dt = float(current_fix["t"] - prev_fix["t"])
        if not np.isfinite(dt) or dt <= 0 or dt > cfg.gap_threshold_s:
            close_block()
            block_items = [item]
            dp = cfg.emission_weight * emission
            back = [np.full(len(g), -1, dtype="int64")]
            continue

        gps_d = float(haversine_m(prev_fix["lon"], prev_fix["lat"],
                                  current_fix["lon"], current_fix["lat"]))
        trans = _transition_scores(prev_g, g, net, gps_d, dt, cfg, cache)
        scores = dp[:, None] + trans
        best_prev = np.argmax(scores, axis=0)
        best_score = scores[best_prev, np.arange(scores.shape[1])]
        if not np.isfinite(best_score).any():
            # Disconnected graph: close the previous chain and restart at this fix.
            close_block()
            block_items = [item]
            dp = cfg.emission_weight * emission
            back = [np.full(len(g), -1, dtype="int64")]
        else:
            dp = best_score + cfg.emission_weight * emission
            back.append(best_prev.astype("int64"))
            block_items.append(item)

    close_block()
    return output


def classifier_top1(candidates: pd.DataFrame, probabilities: Iterable[float]) -> dict[int, int]:
    """Return {fix_idx: arc_id} for the highest-probability candidate per fix."""
    d = candidates[["fix_idx", "arc_id"]].copy()
    d["prob"] = np.asarray(list(probabilities), dtype="float64")
    if len(d) != len(candidates):
        raise ValueError("probabilities must align with candidate rows")
    if "cand_rank" in candidates:
        d["cand_rank"] = candidates["cand_rank"].to_numpy()
    else:
        d["cand_rank"] = 0
    d = d.sort_values(["fix_idx", "prob", "cand_rank"], ascending=[True, False, True], kind="stable")
    d = d.drop_duplicates("fix_idx")
    return dict(zip(d.fix_idx.astype(int), d.arc_id.astype(int)))
