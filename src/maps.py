"""Folium map builders (shared by EDA, model analysis and the API/UI)."""
from __future__ import annotations

from html import escape

import folium
import numpy as np

from .geo import haversine_m
from .network import RoadNetwork
from .plot_style import (CRITICAL, GPS_COLOR, INK, INK_2, NETWORK_COLOR, PARTITION_COLOR, PARTITION_LABEL, PRED_COLOR,
                         ROUTE_COLOR)


def _base_map(lat: float, lon: float, zoom: int = 14) -> folium.Map:
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("Esri.WorldImagery", name="Satellite (Esri)", show=False).add_to(m)
    return m


def _endpoint_markers(m: folium.Map, start, end) -> None:
    """Start/end pins with Font Awesome icons (bundled by folium; the glyphicon set often fails to load)."""
    folium.Marker(start, tooltip="Start", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    folium.Marker(end, tooltip="End", icon=folium.Icon(color="darkred", icon="flag-checkered", prefix="fa")).add_to(m)


def add_legend(m: folium.Map, title: str, items: list[tuple[str, str, str]]) -> None:
    """items: (label, colour, kind) with kind in {'line', 'dot', 'ring', 'dash'}."""
    rows = []
    for label, color, kind in items:
        if kind == "line":
            sw = f'<span style="display:inline-block;width:22px;height:0;border-top:4px solid {color};vertical-align:middle"></span>'
        elif kind == "dash":
            sw = f'<span style="display:inline-block;width:22px;height:0;border-top:3px dashed {color};vertical-align:middle"></span>'
        elif kind == "ring":
            sw = f'<span style="display:inline-block;width:12px;height:12px;border:2px solid {color};border-radius:50%;vertical-align:middle;margin:0 5px"></span>'
        else:
            sw = f'<span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:50%;vertical-align:middle;margin:0 6px"></span>'
        rows.append(f'<div style="margin:3px 0">{sw}&nbsp;<span style="color:{INK_2}">{escape(label)}</span></div>')
    html = (
        '<div style="position:fixed;bottom:28px;left:12px;z-index:9999;background:rgba(252,252,251,0.95);'
        'border:1px solid rgba(11,11,11,0.12);border-radius:8px;padding:10px 12px;'
        'font:12px system-ui,-apple-system,Segoe UI,sans-serif;box-shadow:0 1px 4px rgba(0,0,0,0.12)">'
        f'<div style="font-weight:600;color:{INK};margin-bottom:4px">{escape(title)}</div>{"".join(rows)}</div>'
    )
    m.get_root().html.add_child(folium.Element(html))


def trip_map(trip, net: RoadNetwork, dist_to_route=None, gap_threshold_s: float = 10.0,
             network_buffer_m: float = 150.0, max_points: int = 5000) -> folium.Map:
    """GPS observations, nearby road network and reference route for one trip."""
    tr = trip.track.dropna()
    lon, lat, t = tr["lon"].to_numpy(), tr["lat"].to_numpy(), tr["t"].to_numpy()
    px, py = net.proj.to_xy(lon, lat)
    m = _base_map(float(np.mean(lat)), float(np.mean(lon)))

    # Road network near the track
    stride = max(1, len(px) // 2000)
    near = net.arcs_within(px[::stride], py[::stride], network_buffer_m)
    near_arcs = np.unique(near["arc_id"].to_numpy())
    fg_net = folium.FeatureGroup(name=f"Road network (within {network_buffer_m:.0f} m)")
    segs = net.arc_segments_lonlat(near_arcs)
    if segs:
        folium.PolyLine(segs, color=NETWORK_COLOR, weight=1.5, opacity=0.8).add_to(fg_net)
    fg_net.add_to(m)

    # Reference route
    if trip.has_route:
        fg_route = folium.FeatureGroup(name="Reference route")
        for line in net.route_polylines(trip.route):
            folium.PolyLine(line, color=ROUTE_COLOR, weight=5, opacity=0.85,
                            tooltip="Reference route").add_to(fg_route)
        fg_route.add_to(m)

    # GPS observations
    step = max(1, int(np.ceil(len(lon) / max_points)))
    shown = "all" if step == 1 else f"every {step}th"
    fg_gps = folium.FeatureGroup(name=f"GPS observations ({shown})")
    for i in range(0, len(lon), step):
        tip = f"#{i}  t={t[i]:.0f}s"
        if dist_to_route is not None:
            tip += f"  |  {dist_to_route[i]:.1f} m from route"
        folium.CircleMarker([lat[i], lon[i]], radius=3, color=GPS_COLOR, weight=1, fill=True,
                            fill_color=GPS_COLOR, fill_opacity=0.85, tooltip=tip).add_to(fg_gps)
    fg_gps.add_to(m)

    # Temporal gaps
    dt = np.diff(t)
    gap_idx = np.flatnonzero(dt > gap_threshold_s) + 1
    if len(gap_idx):
        fg_gap = folium.FeatureGroup(name=f"GPS gaps > {gap_threshold_s:.0f} s ({len(gap_idx)})")
        for i in gap_idx:
            jump = haversine_m(lon[i - 1], lat[i - 1], lon[i], lat[i])
            tip = f"Gap of {dt[i - 1]:.0f} s before point #{i} (jump {jump:.0f} m)"
            folium.PolyLine([[lat[i - 1], lon[i - 1]], [lat[i], lon[i]]], color=CRITICAL, weight=2.5,
                            dash_array="6,6", tooltip=tip).add_to(fg_gap)
            folium.CircleMarker([lat[i], lon[i]], radius=8, color=CRITICAL, weight=2.5, fill=False,
                                tooltip=tip).add_to(fg_gap)
        fg_gap.add_to(m)

    _endpoint_markers(m, [lat[0], lon[0]], [lat[-1], lon[-1]])
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])

    items = [("Road network", NETWORK_COLOR, "line"), ("Reference route", ROUTE_COLOR, "line"),
             ("GPS observation", GPS_COLOR, "dot")]
    if len(gap_idx):
        items.append((f"GPS gap > {gap_threshold_s:.0f} s", CRITICAL, "dash"))
    add_legend(m, f"{PARTITION_LABEL.get(trip.partition, trip.partition)} trip {trip.trip_id}", items)
    return m


def _as_polylines(route) -> list[list[list[float]]]:
    """Accept one polyline [[lat, lon], ...] or several [[[lat, lon], ...], ...]."""
    if not route:
        return []
    first = route[0]
    if len(first) and isinstance(first[0], (list, tuple)):
        return [list(map(list, line)) for line in route if len(line) >= 2]
    return [list(map(list, route))] if len(route) >= 2 else []


def prediction_map(gps, predicted_route, title: str, max_points: int = 1500) -> folium.Map:
    """GPS fixes and a predicted route, both as [lat, lon] lists (the API response format).

    Used by the evaluator UI, which only ever receives predictions, never the reference route.
    """
    pts = np.asarray(gps, dtype="float64").reshape(-1, 2)
    lines = _as_polylines(predicted_route)
    all_pts = np.vstack([pts] + [np.asarray(line, dtype="float64") for line in lines]) if lines else pts
    m = _base_map(float(all_pts[:, 0].mean()), float(all_pts[:, 1].mean()))

    # GPS first, predicted route on top, so the prediction stays visible over dense fixes.
    if len(pts):
        step = max(1, int(np.ceil(len(pts) / max_points)))
        shown = "all" if step == 1 else f"every {step}th"
        fg_gps = folium.FeatureGroup(name=f"GPS observations ({shown})")
        for i in range(0, len(pts), step):
            folium.CircleMarker(pts[i].tolist(), radius=2.5, color=GPS_COLOR, weight=1, fill=True,
                                fill_color=GPS_COLOR, fill_opacity=0.7, tooltip=f"fix #{i}").add_to(fg_gps)
        fg_gps.add_to(m)

    if lines:
        fg_route = folium.FeatureGroup(name="Predicted route")
        for line in lines:
            folium.PolyLine(line, color=PRED_COLOR, weight=5, opacity=0.9, tooltip="Predicted route").add_to(fg_route)
        fg_route.add_to(m)

    if len(pts):
        _endpoint_markers(m, pts[0].tolist(), pts[-1].tolist())

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[float(all_pts[:, 0].min()), float(all_pts[:, 1].min())],
                  [float(all_pts[:, 0].max()), float(all_pts[:, 1].max())]])
    add_legend(m, title, [("Predicted route", PRED_COLOR, "line"), ("GPS observation", GPS_COLOR, "dot")])
    return m


def overview_map(trips_df) -> folium.Map:
    """One marker per trip at its GPS centroid, coloured by partition."""
    m = _base_map(float(trips_df["centroid_lat"].mean()), float(trips_df["centroid_lon"].mean()), zoom=2)
    for part, grp in trips_df.groupby("partition"):
        fg = folium.FeatureGroup(name=f"{PARTITION_LABEL.get(part, part)} ({len(grp)} trips)")
        for _, r in grp.iterrows():
            tip = (f"{PARTITION_LABEL.get(part, part)} trip {r['trip_id']}<br>"
                   f"{int(r['n_points'])} GPS points, {r['duration_min']:.0f} min, {r['path_length_km']:.1f} km<br>"
                   f"network: {int(r['n_nodes'])} nodes / {int(r['n_arcs'])} arcs")
            folium.CircleMarker([r["centroid_lat"], r["centroid_lon"]], radius=7, color="#fcfcfb", weight=2,
                                fill=True, fill_color=PARTITION_COLOR.get(part, GPS_COLOR), fill_opacity=0.95,
                                tooltip=folium.Tooltip(tip)).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[trips_df["centroid_lat"].min(), trips_df["centroid_lon"].min()],
                  [trips_df["centroid_lat"].max(), trips_df["centroid_lon"].max()]])
    add_legend(m, "Trip locations", [(PARTITION_LABEL[p], PARTITION_COLOR[p], "dot")
                                     for p in PARTITION_COLOR if p in set(trips_df["partition"])])
    return m
