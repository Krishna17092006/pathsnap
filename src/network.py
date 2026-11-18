"""Road-network geometry and connectivity for a single trip.

Each trip ships its own road network, so arc ids are only meaningful inside
that trip. RoadNetwork projects the nodes into the trip's local metric frame
and offers:

* arc lengths (haversine, metres)
* a spatial index over densified arc samples, used to find every arc within
  a radius of a point (the basis for candidate generation)
* connectivity statistics (isolated nodes, weak/strong components)
* reference-route helpers (breaks, polylines for plotting)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .geo import LocalProjection, haversine_m, point_segment_distance


class RoadNetwork:
    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame, proj: LocalProjection,
                 sample_spacing_m: float = 10.0):
        self.proj = proj
        self.n_nodes = len(nodes)
        self.n_arcs = len(arcs)
        self.node_lon = nodes["lon"].to_numpy(dtype="float64")
        self.node_lat = nodes["lat"].to_numpy(dtype="float64")
        self.u = arcs["u"].to_numpy(dtype="int64")
        self.v = arcs["v"].to_numpy(dtype="int64")
        bad = (self.u < 0) | (self.u >= self.n_nodes) | (self.v < 0) | (self.v >= self.n_nodes)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} arcs reference node ids outside 0..{self.n_nodes - 1}")

        self.node_x, self.node_y = proj.to_xy(self.node_lon, self.node_lat)
        self.ax, self.ay = self.node_x[self.u], self.node_y[self.u]
        self.bx, self.by = self.node_x[self.v], self.node_y[self.v]
        self.length_m = haversine_m(self.node_lon[self.u], self.node_lat[self.u],
                                    self.node_lon[self.v], self.node_lat[self.v])
        self.sample_spacing_m = float(sample_spacing_m)
        self._tree: cKDTree | None = None
        self._sample_arc: np.ndarray | None = None

    # ------------------------------------------------------------------ spatial index
    def _build_index(self) -> None:
        """Densify every arc into points at most `sample_spacing_m` apart and index them.

        Any point on an arc is then within spacing/2 of some sample, so a
        ball query of radius r + spacing/2 never misses an arc within r.
        """
        seg_len = np.hypot(self.bx - self.ax, self.by - self.ay)
        n_int = np.maximum(1, np.ceil(seg_len / self.sample_spacing_m).astype("int64"))
        counts = n_int + 1
        arc_idx = np.repeat(np.arange(self.n_arcs), counts)
        first = np.repeat(np.cumsum(counts) - counts, counts)
        frac = (np.arange(counts.sum()) - first) / np.repeat(n_int, counts)
        sx = self.ax[arc_idx] + frac * (self.bx - self.ax)[arc_idx]
        sy = self.ay[arc_idx] + frac * (self.by - self.ay)[arc_idx]
        self._tree = cKDTree(np.column_stack([sx, sy]))
        self._sample_arc = arc_idx

    def arcs_within(self, px, py, radius_m: float) -> pd.DataFrame:
        """All (point, arc) pairs with exact point-to-arc distance <= radius_m.

        Returns columns: point_idx, arc_id, dist_m, t (projection fraction along the arc).
        """
        if self._tree is None:
            self._build_index()
        px = np.asarray(px, dtype="float64")
        py = np.asarray(py, dtype="float64")
        empty = pd.DataFrame({"point_idx": pd.Series(dtype="int64"), "arc_id": pd.Series(dtype="int64"),
                              "dist_m": pd.Series(dtype="float64"), "t": pd.Series(dtype="float64")})
        if len(px) == 0:
            return empty
        hits = self._tree.query_ball_point(np.column_stack([px, py]),
                                           radius_m + self.sample_spacing_m / 2, workers=-1)
        lens = np.fromiter((len(h) for h in hits), dtype="int64", count=len(hits))
        if lens.sum() == 0:
            return empty
        point_idx = np.repeat(np.arange(len(px)), lens)
        samples = np.concatenate([np.asarray(h, dtype="int64") for h in hits if h])
        key = np.unique(point_idx * self.n_arcs + self._sample_arc[samples])
        point_idx, arc_id = key // self.n_arcs, key % self.n_arcs
        dist, t = point_segment_distance(px[point_idx], py[point_idx],
                                         self.ax[arc_id], self.ay[arc_id], self.bx[arc_id], self.by[arc_id])
        keep = dist <= radius_m
        return pd.DataFrame({"point_idx": point_idx[keep], "arc_id": arc_id[keep],
                             "dist_m": dist[keep], "t": t[keep]})

    def distance_to_arcs(self, px, py, arc_ids, chunk: int = 2000):
        """Exact distance from each point to the nearest arc in `arc_ids` (brute force, chunked).

        Returns (min_dist_m, nearest_arc_id). Intended for small arc sets such as a route.
        """
        arc_ids = np.asarray(arc_ids, dtype="int64")
        px = np.asarray(px, dtype="float64")
        py = np.asarray(py, dtype="float64")
        ax, ay, bx, by = (a[arc_ids][None, :] for a in (self.ax, self.ay, self.bx, self.by))
        best_d = np.empty(len(px))
        best_a = np.empty(len(px), dtype="int64")
        for s in range(0, len(px), chunk):
            d, _ = point_segment_distance(px[s:s + chunk, None], py[s:s + chunk, None], ax, ay, bx, by)
            j = np.argmin(d, axis=1)
            best_d[s:s + chunk] = d[np.arange(len(j)), j]
            best_a[s:s + chunk] = arc_ids[j]
        return best_d, best_a

    # ------------------------------------------------------------------ connectivity
    def connectivity_stats(self) -> dict:
        n = self.n_nodes
        adj = coo_matrix((np.ones(self.n_arcs), (self.u, self.v)), shape=(n, n)).tocsr()
        used = np.zeros(n, dtype=bool)
        used[self.u] = True
        used[self.v] = True
        n_weak, weak_lab = connected_components(adj, directed=True, connection="weak")
        n_strong, strong_lab = connected_components(adj, directed=True, connection="strong")
        weak_sizes = np.bincount(weak_lab[used]) if used.any() else np.array([0])
        strong_sizes = np.bincount(strong_lab)
        largest_scc = int(np.argmax(strong_sizes))
        out_deg = np.bincount(self.u, minlength=n)
        in_deg = np.bincount(self.v, minlength=n)
        self.largest_scc_mask = strong_lab == largest_scc
        return {
            "isolated_nodes": int((~used).sum()),
            "weak_components": int((weak_sizes > 0).sum()),
            "largest_wcc_share": float(weak_sizes.max() / max(used.sum(), 1)),
            "strong_components": int(n_strong),
            "largest_scc_share": float(strong_sizes.max() / n),
            "dead_end_nodes": int(((out_deg == 0) & (in_deg > 0)).sum()),
            "source_only_nodes": int(((in_deg == 0) & (out_deg > 0)).sum()),
            "mean_out_degree": float(out_deg[used].mean()) if used.any() else 0.0,
        }

    # ------------------------------------------------------------------ route helpers
    def route_breaks(self, route) -> np.ndarray:
        """Indices i where route[i] does not end at the node where route[i+1] starts."""
        route = np.asarray(route, dtype="int64")
        if len(route) < 2:
            return np.array([], dtype="int64")
        return np.flatnonzero(self.v[route[:-1]] != self.u[route[1:]])

    def route_polylines(self, route) -> list[list[tuple[float, float]]]:
        """Route as (lat, lon) polylines, split wherever consecutive arcs do not connect."""
        route = np.asarray(route, dtype="int64")
        if len(route) == 0:
            return []
        breaks = set(self.route_breaks(route).tolist())
        lines, cur = [], [(self.node_lat[self.u[route[0]]], self.node_lon[self.u[route[0]]])]
        for i, a in enumerate(route):
            cur.append((self.node_lat[self.v[a]], self.node_lon[self.v[a]]))
            if i in breaks:
                lines.append(cur)
                nxt = route[i + 1]
                cur = [(self.node_lat[self.u[nxt]], self.node_lon[self.u[nxt]])]
        lines.append(cur)
        return lines

    def arc_segments_lonlat(self, arc_ids, dedupe_two_way: bool = True) -> list[list[tuple[float, float]]]:
        """(lat, lon) pairs for drawing arcs; optionally draw each two-way road once."""
        arc_ids = np.asarray(arc_ids, dtype="int64")
        if dedupe_two_way and len(arc_ids):
            lo = np.minimum(self.u[arc_ids], self.v[arc_ids])
            hi = np.maximum(self.u[arc_ids], self.v[arc_ids])
            _, first = np.unique(lo * self.n_nodes + hi, return_index=True)
            arc_ids = arc_ids[np.sort(first)]
        return [[(self.node_lat[self.u[a]], self.node_lon[self.u[a]]),
                 (self.node_lat[self.v[a]], self.node_lon[self.v[a]])] for a in arc_ids]
