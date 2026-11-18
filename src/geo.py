"""Geodesic helpers.

Longitude/latitude are angles, so Euclidean distance in degrees is not a
physical distance (1 deg of longitude shrinks with cos(latitude)). We use:

* haversine_m: great-circle distance on a sphere, used for reporting lengths.
* LocalProjection: an equirectangular tangent-plane projection centred on each
  trip, giving x/y in metres for vector geometry (point-to-segment distance,
  KD-trees). Each trip covers at most tens of km, and the EDA measures the
  projection error against haversine per trip to justify this choice.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius (IUGG)


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(a, dtype="float64")) for a in (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lon1, lat1, lon2, lat2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north in [0, 360)."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(a, dtype="float64")) for a in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


class LocalProjection:
    """Equirectangular projection around (lon0, lat0): x east, y north, metres."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self._kx = EARTH_RADIUS_M * np.cos(np.radians(self.lat0)) * np.pi / 180.0
        self._ky = EARTH_RADIUS_M * np.pi / 180.0

    @classmethod
    def from_points(cls, lon, lat) -> "LocalProjection":
        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
        ok = np.isfinite(lon) & np.isfinite(lat)
        return cls(np.mean(lon[ok]), np.mean(lat[ok]))

    def to_xy(self, lon, lat):
        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
        return (lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky

    def to_lonlat(self, x, y):
        return np.asarray(x) / self._kx + self.lon0, np.asarray(y) / self._ky + self.lat0


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (broadcasting, projected metres).

    Returns (dist, t) where t in [0, 1] is the position of the closest point
    along AB (0 = A, 1 = B). Zero-length segments return the distance to A.
    """
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 0, l2, 1.0)
    t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * dx
    cy = ay + t * dy
    return np.hypot(px - cx, py - cy), t
