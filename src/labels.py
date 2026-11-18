"""Per-fix labels derived from the reference route.

The reference route lists arcs in driving order but never says which GPS fix
belongs to which arc. Labelling each fix with its plain nearest route arc fails
wherever the route passes the same place twice (U-turns, loops, revisited
arcs, close parallel sections). Instead every fix gets a *position along the
route* through an order-constrained dynamic programme (a monotone alignment):

    cost[0, j] = c(0, j)
    cost[i, j] = c(i, j) + min_{j' <= j} cost[i-1, j']
    c(i, j)    = min(distance(fix i, arc route[j]), LABEL_COST_CAP_M)

so the matched route position never moves backwards in time. The optimal
positions are recovered by backtracking. The cap keeps long stretches that the
route does not cover (fixes hundreds of metres away) from distorting the
alignment of everything else.

A fix whose matched arc lies farther than LABEL_MAX_DIST_M is marked invalid:
there the reference route does not describe where the vehicle was, so the fix
has no reliable label. Invalid fixes are excluded from training and metrics but
are kept for inference (the HMM still has to decode them).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LABEL_COST_CAP_M, LABEL_MAX_DIST_M
from .geo import point_segment_distance
from .network import RoadNetwork


@dataclass
class FixLabels:
    route_pos: np.ndarray     # index into the route for every fix (-1 if the trip has no route)
    true_arc: np.ndarray      # matched arc id, -1 where the fix is not validly labelled
    match_dist_m: np.ndarray  # distance from the fix to its matched route arc
    fix_valid: np.ndarray     # True where the label is reliable


def match_to_route(px, py, net: RoadNetwork, route, cap_m: float = LABEL_COST_CAP_M, chunk: int = 1024):
    """Monotone alignment of fixes to route positions. Returns (route_pos, match_dist_m)."""
    route = np.asarray(route, dtype="int64")
    px = np.asarray(px, dtype="float64")
    py = np.asarray(py, dtype="float64")
    n, m = len(px), len(route)
    ax, ay = net.ax[route][None, :], net.ay[route][None, :]
    bx, by = net.bx[route][None, :], net.by[route][None, :]
    idx = np.arange(m)
    back = np.zeros((n, m), dtype=np.int16 if m < 2 ** 15 else np.int32)

    prev = None
    for s in range(0, n, chunk):
        d, _ = point_segment_distance(px[s:s + chunk, None], py[s:s + chunk, None], ax, ay, bx, by)
        c = np.minimum(d, cap_m)
        for k in range(c.shape[0]):
            if prev is None:
                prev = c[k].astype("float64")
                continue
            run_min = np.minimum.accumulate(prev)
            # latest index j' <= j achieving the running minimum
            back[s + k] = np.maximum.accumulate(np.where(prev <= run_min, idx, 0))
            prev = c[k] + run_min

    pos = np.empty(n, dtype="int64")
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]

    arcs = route[pos]
    dist, _ = point_segment_distance(px, py, net.ax[arcs], net.ay[arcs], net.bx[arcs], net.by[arcs])
    return pos, dist


def make_labels(px, py, net: RoadNetwork, route, max_dist_m: float = LABEL_MAX_DIST_M) -> FixLabels:
    n = len(px)
    if route is None or len(route) == 0 or n == 0:
        return FixLabels(np.full(n, -1), np.full(n, -1), np.full(n, np.nan), np.zeros(n, dtype=bool))
    route = np.asarray(route, dtype="int64")
    pos, dist = match_to_route(px, py, net, route)
    valid = dist <= max_dist_m
    true_arc = np.where(valid, route[pos], -1)
    return FixLabels(pos, true_arc, dist, valid)
