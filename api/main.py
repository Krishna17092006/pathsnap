"""PathSnap FastAPI backend (Laptop 1).

Serves the real end-to-end pipeline from src/pipeline.py: candidate generation, feature
engineering, classifier probabilities (artifacts/model.joblib, the tuned XGBoost) and
HMM/Viterbi decoding. Nothing here re-implements map matching.

Run from the project root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    PATHSNAP_TRIPS_DIR=data/student_test uvicorn api.main:app --host 0.0.0.0 --port 8000   # rehearsal

Environment
    PATHSNAP_TRIPS_DIR       trip folders <dir>/<trip_id>/<trip_id>.{track,nodes,arcs[,route]}
                             (default: data/eval; relative paths are resolved from the project root)
    PATHSNAP_ARTIFACTS_DIR   artifact folder (default: artifacts/)
    PATHSNAP_CORS_ORIGINS    comma-separated allowed origins for the UI (default: *)

Endpoints
    GET  /health               {"status": "ok", "mode": "real", ...}
    GET  /trips                {"trips": [...]}
    POST /predict/{trip_id}    prediction only. The trip is loaded WITHOUT its .route file, and no
                               reference-based metric is computed.
    POST /evaluate/{trip_id}   trusted server-side scoring (src.pipeline.score_trip). Prediction is
                               still made without the route; the route is read only for scoring, and
                               only aggregate metrics are returned (never the route or per-fix labels).

Startup fails if the artifacts are missing or invalid. There is no mock fallback.
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import ARTIFACTS_DIR
from src.data_io import PROJECT_ROOT, list_trips, load_trip, read_raw, trip_file, trip_from_raw
from src.export import read_artifacts
from src.pipeline import load_artifacts, predict_trip, score_trip
from src.route_metrics import route_continuity

REQUIRED_ARTIFACTS = ("model.joblib", "features.json", "hmm.json", "metadata.json")
TRIP_ID = re.compile(r"^\d{8}$")
SCORE_KEYS = ("top1_classifier", "top1_hmm", "route_continuity_clf", "route_continuity_hmm",
              "route_precision", "route_recall", "route_f1", "n_fixes", "n_valid_fixes")

log = logging.getLogger("pathsnap.api")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


# ------------------------------------------------------------------------------ configuration
def _resolve(path_str: str | None, default: Path) -> Path:
    if not path_str:
        return default
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_validated_artifacts(artifact_dir: Path) -> tuple[dict, dict]:
    """Check the four shared artifact files, verify the checksum, then load them for the pipeline.

    Raises RuntimeError with a clear message instead of starting in a degraded or mock mode.
    """
    missing = [f for f in REQUIRED_ARTIFACTS if not (artifact_dir / f).exists()]
    if missing:
        raise RuntimeError(f"artifacts missing in {artifact_dir}: {missing}. "
                           f"Run `python -m src.export` to create them.")
    try:
        verified = read_artifacts(artifact_dir)          # checksum + feature list
    except Exception as exc:
        raise RuntimeError(f"invalid artifacts in {artifact_dir}: {exc}") from exc
    art = load_artifacts(artifact_dir)                    # format used by predict_trip
    if art.get("model_bundle") is None or not str(art.get("model_path", "")).endswith("model.joblib"):
        raise RuntimeError("load_artifacts did not load artifacts/model.joblib")
    return art, verified.metadata


def trip_dir(trips_dir: Path, trip_id: str) -> Path:
    """Validate a trip id (8 digits; this also prevents path traversal) and return its folder."""
    if not TRIP_ID.match(trip_id):
        raise HTTPException(status_code=404, detail=f"unknown trip '{trip_id}'")
    d = trips_dir / trip_id
    if not d.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown trip '{trip_id}'")
    return d


def load_inference_trip(trips_dir: Path, trip_id: str):
    """Load a trip for prediction WITHOUT opening its .route file.

    Reuses src.data_io (read_raw + trip_from_raw). The route frame is empty, so trip.route is None.
    """
    partition, data_dir = trips_dir.name, trips_dir.parent
    raw = {k: read_raw(trip_file(trip_id, partition, k, data_dir), k) for k in ("track", "nodes", "arcs")}
    raw["route"] = pd.DataFrame({"arc_id": pd.Series(dtype="float64")})
    trip = trip_from_raw(trip_id, partition, raw)
    if trip.track.empty:
        raise ValueError(f"{trip_id}: .track is missing or empty")
    if trip.nodes.empty or trip.arcs.empty:
        raise ValueError(f"{trip_id}: road network (.nodes/.arcs) is missing or empty")
    return trip


# ------------------------------------------------------------------------------ JSON helpers
def _num(x, nd: int = 3):
    """Plain JSON number (None for NaN/inf)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _latlon(points, nd: int = 6) -> list[list[float]]:
    return [[round(float(lat), nd), round(float(lon), nd)] for lat, lon in points]


def _gps(trip) -> list[list[float]]:
    tr = trip.track
    ok = np.isfinite(tr["lon"]) & np.isfinite(tr["lat"])
    return _latlon(zip(tr.loc[ok, "lat"], tr.loc[ok, "lon"]))


def _prediction_metrics(result, art: dict) -> dict:
    """Metrics that need no reference route: computed from the predictions and the road network."""
    fixes, net = result.fixes, result._network
    out = {"n_candidates": int(result.diagnostics.get("n_candidates", 0)),
           "n_unmatched_fixes": int(result.diagnostics.get("n_unmatched_fixes", 0)),
           "n_route_arcs": len(result.route_arcs)}
    if net is not None and len(fixes) > 1:
        gap = art["decoder_config"].gap_threshold_s
        out["route_continuity_clf"] = _num(route_continuity(fixes, fixes["arc_clf"], net, gap), 4)
        out["route_continuity_hmm"] = _num(route_continuity(fixes, fixes["arc_hmm"], net, gap), 4)
    return out


def _timing(diag: dict, request_ms: float) -> dict:
    return {"candidates": _num(diag.get("candidate_generation_ms"), 1),
            "model": _num(diag.get("model_ms"), 1),     # feature engineering + classifier probabilities
            "hmm": _num(diag.get("hmm_ms"), 1),
            "pipeline": _num(diag.get("total_ms"), 1),
            "total": _num(request_ms, 1)}


# ------------------------------------------------------------------------------ app
@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = _resolve(os.environ.get("PATHSNAP_ARTIFACTS_DIR"), ARTIFACTS_DIR)
    trips_dir = _resolve(os.environ.get("PATHSNAP_TRIPS_DIR"), PROJECT_ROOT / "data" / "eval")
    art, metadata = load_validated_artifacts(artifact_dir)     # once per process
    app.state.artifacts = art
    app.state.metadata = metadata
    app.state.trips_dir = trips_dir
    log.info("artifacts loaded once from %s | model=%s (%s) | hmm beta=%s | trips dir=%s (%s)",
             artifact_dir, metadata.get("model"), metadata.get("source"), art["decoder_config"].beta,
             trips_dir, "exists" if trips_dir.is_dir() else "MISSING")
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    meta = app.state.metadata
    return {"status": "ok", "mode": "real", "model": meta.get("model"), "model_source": meta.get("source")}


@app.get("/trips")
def trips() -> dict:
    d = app.state.trips_dir
    if not d.is_dir():
        raise HTTPException(status_code=503, detail=f"trips directory not found: {d.name}")
    return {"trips": [t for t in list_trips(d.name, d.parent) if TRIP_ID.match(t)]}


def _run_prediction(trip_id: str):
    trips_dir: Path = app.state.trips_dir
    trip_dir(trips_dir, trip_id)
    try:
        trip = load_inference_trip(trips_dir, trip_id)
    except (ValueError, KeyError) as exc:
        log.warning("trip %s rejected: %s", trip_id, exc)
        raise HTTPException(status_code=422, detail=f"malformed trip data: {exc}") from exc
    try:
        result = predict_trip(trip, app.state.artifacts)
    except Exception as exc:
        log.exception("inference failed for trip %s", trip_id)
        raise HTTPException(status_code=500, detail="inference failed; see the server log") from exc
    return trip, result


@app.post("/predict/{trip_id}")
def predict(trip_id: str) -> dict:
    t0 = time.perf_counter()
    trip, result = _run_prediction(trip_id)
    metrics = _prediction_metrics(result, app.state.artifacts)
    total_ms = (time.perf_counter() - t0) * 1000.0
    timing = _timing(result.diagnostics, total_ms)
    log.info("predict trip=%s n_fixes=%d candidates=%d route_arcs=%d hmm_ms=%s total_ms=%s",
             trip_id, len(trip.track), metrics["n_candidates"], metrics["n_route_arcs"],
             timing["hmm"], timing["total"])
    return {
        "trip_id": trip_id,
        "n_fixes": int(len(trip.track)),
        "gps": _gps(trip),
        "predicted_route": [_latlon(line) for line in result.route_polyline],
        "predicted_arcs": [int(a) for a in result.route_arcs],
        "metrics": metrics,
        "timing_ms": timing,
        "mode": "real",
    }


@app.post("/evaluate/{trip_id}")
def evaluate(trip_id: str) -> dict:
    """Server-side accuracy against the private reference route (aggregate numbers only)."""
    t0 = time.perf_counter()
    trip, result = _run_prediction(trip_id)              # prediction made without the route
    try:
        reference = load_trip(trip_id, app.state.trips_dir.name, app.state.trips_dir.parent)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"malformed trip data: {exc}") from exc
    if not reference.has_route:
        raise HTTPException(status_code=409, detail=f"no reference route available for trip '{trip_id}'")
    scored = score_trip(result, replace(trip, route=reference.route))
    metrics = {k: (_num(scored[k], 4) if k.startswith(("top1", "route")) else int(scored[k]))
               for k in SCORE_KEYS if k in scored}
    total_ms = (time.perf_counter() - t0) * 1000.0
    log.info("evaluate trip=%s n_fixes=%d top1_clf=%s top1_hmm=%s continuity_hmm=%s total_ms=%.1f",
             trip_id, len(trip.track), metrics.get("top1_classifier"), metrics.get("top1_hmm"),
             metrics.get("route_continuity_hmm"), total_ms)
    return {"trip_id": trip_id, "metrics": metrics, "timing_ms": _timing(result.diagnostics, total_ms),
            "mode": "real"}
