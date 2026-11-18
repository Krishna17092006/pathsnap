"""End-to-end PathSnap inference pipeline (no reference route required at inference)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from .config import CANDIDATE_RADIUS_M, ARTIFACTS_DIR
from .data_io import Trip
from .features import FEATURE_COLUMNS, add_features
from .geo import LocalProjection
from .hmm import DecoderConfig, classifier_top1, viterbi_decode
from .labels import make_labels
from .network import RoadNetwork
from .route_metrics import compare_predictions


@dataclass
class PredictionResult:
    fixes: pd.DataFrame
    route_arcs: list[int]
    route_polyline: list[list[tuple[float, float]]]
    diagnostics: dict = field(default_factory=dict)
    _network: RoadNetwork | None = field(default=None, repr=False)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_artifacts(artifact_dir: str | Path = ARTIFACTS_DIR) -> dict:
    """Load the selected classifier, ordered features and HMM decoder settings.

    Supports the team's shared `model.joblib` and Julia's standalone
    `mlp_model.joblib` bundle. It never silently replaces the shared model.
    """
    root = Path(artifact_dir)
    model_path = root / "model.joblib"
    if not model_path.exists():
        model_path = root / "mlp_model.joblib"
    model_bundle = joblib.load(model_path) if model_path.exists() else None

    feature_columns = _read_json(root / "features.json", None)
    if isinstance(model_bundle, dict):
        feature_columns = (feature_columns or model_bundle.get("feature_columns")
                           or model_bundle.get("features"))
    if not feature_columns:
        feature_columns = list(FEATURE_COLUMNS)

    hmm_values = _read_json(root / "hmm.json", {})
    allowed = set(DecoderConfig.__dataclass_fields__)
    decoder = DecoderConfig(**{k: v for k, v in hmm_values.items() if k in allowed})
    return {"model_bundle": model_bundle, "feature_columns": list(feature_columns),
            "decoder_config": decoder, "artifact_dir": str(root),
            "model_path": str(model_path) if model_path.exists() else None,
            "model_name": _model_name(root, model_path, model_bundle)}


def _model_name(root: Path, model_path: Path, model_bundle) -> str:
    """Short name of the classifier behind the HMM emissions, used to name result folders."""
    if model_bundle is None:
        return "distance_gaussian"
    if model_path.name == "mlp_model.joblib":
        return "mlp"
    return str(_read_json(root / "metadata.json", {}).get("model") or "model")


def _model_probabilities(feature_frame: pd.DataFrame, art: dict) -> np.ndarray | None:
    bundle = art.get("model_bundle")
    if bundle is None:
        return None
    cols = art.get("feature_columns", FEATURE_COLUMNS)
    X = feature_frame.reindex(columns=cols).replace([np.inf, -np.inf], np.nan)
    if isinstance(bundle, dict):
        model = bundle.get("model", bundle.get("estimator", bundle.get("pipeline")))
        if model is None:
            raise ValueError("model artifact dictionary has no 'model', 'estimator' or 'pipeline'")
        medians = bundle.get("medians", {})
        if medians:
            X = X.fillna(pd.Series(medians))
        else:
            X = X.fillna(0.0)
        scaler = bundle.get("scaler")
        values = X.to_numpy(dtype="float64")
        if scaler is not None:
            # Preserve fitted feature names when the scaler was trained on a DataFrame.
            values = scaler.transform(X if hasattr(scaler, "feature_names_in_") else values)
        return np.asarray(model.predict_proba(values)[:, 1], dtype="float64")
    # sklearn Pipeline or estimator that performs its own preprocessing
    # (logreg, knn, gaussian_nb, random_forest, xgboost from src.model_zoo / src.export).
    return np.asarray(bundle.predict_proba(model_input(bundle, X))[:, 1], dtype="float64")


def _handles_missing(model) -> bool:
    """True when the model imputes NaN itself (e.g. the tree pipelines' SimpleImputer)."""
    steps = getattr(model, "steps", None)
    first = steps[0][1] if steps else model
    return hasattr(first, "missing_values") or type(first).__name__.startswith(("XGB", "HistGradient"))


def model_input(model, X: pd.DataFrame):
    """Shape candidate features the way `model` was fitted.

    - Fitted on a DataFrame (e.g. the Optuna-tuned model from src.tune): pass a
      DataFrame in the fitted column order, so columns are matched by name.
    - Fitted on a NumPy array (src.train / src.export): pass a float array.
    - Missing values: left as NaN when the model has its own imputer, so it applies
      the training medians; otherwise filled with 0.0 so models without an imputer
      (logistic regression, KNN, Naive Bayes) do not fail.
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    if not _handles_missing(model):
        X = X.fillna(0.0)
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        missing = [c for c in names if c not in X.columns]
        if missing:
            raise ValueError(f"model expects feature columns that are missing: {missing}")
        return X.loc[:, list(names)].astype("float64")
    return X.to_numpy(dtype="float64")


def _build_inference_candidates(trip: Trip, radius_m: float):
    track = trip.track
    ok = (np.isfinite(track["lon"]) & np.isfinite(track["lat"]) & np.isfinite(track["t"])
          & track["lat"].between(-90, 90) & track["lon"].between(-180, 180)).to_numpy()
    raw_indices = np.flatnonzero(ok)
    valid_track = track.iloc[raw_indices].reset_index(drop=True).copy()
    empty_fixes = pd.DataFrame({"fix_idx": np.arange(len(track), dtype="int64"),
                                "t": track["t"].to_numpy(dtype="float64"),
                                "lon": track["lon"].to_numpy(dtype="float64"),
                                "lat": track["lat"].to_numpy(dtype="float64")})
    empty_cand = pd.DataFrame(columns=["trip_id", "partition", "fix_idx", "t", "lon", "lat",
                                       "arc_id", "dist_m", "t_along", "cand_rank", "n_cand"])
    if valid_track.empty or trip.nodes.empty or trip.arcs.empty:
        return None, empty_cand, empty_fixes, raw_indices

    proj = LocalProjection.from_points(valid_track["lon"].to_numpy(), valid_track["lat"].to_numpy())
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    px, py = proj.to_xy(valid_track["lon"].to_numpy(), valid_track["lat"].to_numpy())
    cand = net.arcs_within(px, py, radius_m).rename(columns={"point_idx": "fix_idx", "t": "t_along"})
    if cand.empty:
        return net, empty_cand, empty_fixes, raw_indices
    cand = cand.sort_values(["fix_idx", "dist_m", "arc_id"], kind="stable").reset_index(drop=True)
    cand["cand_rank"] = cand.groupby("fix_idx").cumcount().astype("int32")
    counts = np.bincount(cand["fix_idx"].to_numpy(dtype="int64"), minlength=len(valid_track))
    cand["n_cand"] = counts[cand["fix_idx"].to_numpy(dtype="int64")].astype("int32")
    local_fix = cand["fix_idx"].to_numpy(dtype="int64")
    cand["t"] = valid_track["t"].to_numpy(dtype="float64")[local_fix]
    cand["lon"] = valid_track["lon"].to_numpy(dtype="float64")[local_fix]
    cand["lat"] = valid_track["lat"].to_numpy(dtype="float64")[local_fix]
    cand["trip_id"] = trip.trip_id
    cand["partition"] = trip.partition

    # Features expect fix_idx to index the Trip.track passed in, so use the
    # compact valid-only track while features are built, then restore raw indices.
    feature_trip = SimpleNamespace(track=valid_track, trip_id=trip.trip_id, partition=trip.partition)
    featured = add_features(cand, feature_trip, net)
    featured["_local_fix_idx"] = featured["fix_idx"].to_numpy(dtype="int64")
    featured["fix_idx"] = raw_indices[featured["_local_fix_idx"].to_numpy(dtype="int64")]
    return net, featured, empty_fixes, raw_indices


def predict_trip(trip: Trip, art: dict | None = None, radius_m: float = CANDIDATE_RADIUS_M) -> PredictionResult:
    """Predict road arc for each GPS fix and reconstruct a route without reading `.route`."""
    started = time.perf_counter()
    art = art or load_artifacts()
    candidate_started = time.perf_counter()
    net, cand, fixes, valid_indices = _build_inference_candidates(trip, radius_m)
    candidate_ms = (time.perf_counter() - candidate_started) * 1000.0
    if net is None or cand.empty:
        fixes["arc_clf"] = -1
        fixes["prob_clf"] = np.nan
        fixes["arc_hmm"] = -1
        return PredictionResult(fixes, [], [], {"n_candidates": 0, "n_fixes": int(len(fixes)),
            "n_unmatched_fixes": int(len(fixes)), "candidate_generation_ms": candidate_ms,
            "model_ms": 0.0, "hmm_ms": 0.0,
            "total_ms": (time.perf_counter() - started) * 1000.0}, net)

    model_started = time.perf_counter()
    prob = _model_probabilities(cand, art)
    if prob is None:
        # Distance-only Gaussian fallback: exp(-0.5 * (distance / sigma)^2).
        prob = np.exp(-0.5 * np.square(cand["dist_m"].to_numpy(dtype="float64") /
                                       art["decoder_config"].sigma))
    model_ms = (time.perf_counter() - model_started) * 1000.0
    cand = cand.copy()
    cand["prob"] = np.clip(prob, 1e-6, 1.0)
    clf_map = classifier_top1(cand, prob)
    clf_full = np.full(len(fixes), -1, dtype="int64")
    prob_full = np.full(len(fixes), np.nan, dtype="float64")
    fix_pos = {int(fid): i for i, fid in enumerate(fixes.fix_idx)}
    for fix_id, arc_id in clf_map.items():
        clf_full[fix_pos[int(fix_id)]] = int(arc_id)
        rows = cand[cand.fix_idx == fix_id]
        chosen = rows.loc[rows.arc_id == arc_id, "prob"].iloc[0]
        prob_full[fix_pos[int(fix_id)]] = float(chosen)

    # HMM consumes classifier probabilities when available; without a model it
    # uses the distance Gaussian specified in TEAM.md.
    hmm_started = time.perf_counter()
    hmm_prob = prob if art.get("model_bundle") is not None else None
    hmm_full = viterbi_decode(cand, fixes, net, probabilities=hmm_prob,
                              config=art["decoder_config"])
    hmm_ms = (time.perf_counter() - hmm_started) * 1000.0
    fixes["arc_clf"] = clf_full
    fixes["prob_clf"] = prob_full
    fixes["arc_hmm"] = hmm_full
    route_arcs = []
    for arc in hmm_full:
        if arc >= 0 and (not route_arcs or route_arcs[-1] != int(arc)):
            route_arcs.append(int(arc))
    route_polyline = net.route_polylines(route_arcs) if route_arcs else []
    diagnostics = {"n_candidates": int(len(cand)), "n_fixes": int(len(fixes)),
                   "n_unmatched_fixes": int((hmm_full < 0).sum()),
                   "gap_threshold_s": float(art["decoder_config"].gap_threshold_s),
                   "candidate_generation_ms": candidate_ms, "model_ms": model_ms,
                   "hmm_ms": hmm_ms,
                   "total_ms": (time.perf_counter() - started) * 1000.0}
    return PredictionResult(fixes, route_arcs, route_polyline, diagnostics, net)


def score_trip(pred: PredictionResult, trip: Trip) -> dict:
    """Score predictions against `.route`; call only in the trusted evaluator/server.

    `predict_trip` never accesses the reference route. This separate function is
    the only place in this module that reads it.
    """
    if trip.route is None or len(trip.route) == 0:
        return {"top1_classifier": float("nan"), "top1_hmm": float("nan"),
                "route_continuity_clf": float("nan"), "route_continuity_hmm": float("nan"),
                "route_precision": float("nan"), "route_recall": float("nan"),
                "route_f1": float("nan"), "n_fixes": int(len(trip.track)), "n_valid_fixes": 0}

    pred = pred
    track = trip.track
    ok = (np.isfinite(track["lon"]) & np.isfinite(track["lat"]) & np.isfinite(track["t"])
          & track["lat"].between(-90, 90) & track["lon"].between(-180, 180)).to_numpy()
    idx = np.flatnonzero(ok)
    net = pred._network
    if net is None or len(idx) == 0:
        return {"top1_classifier": float("nan"), "top1_hmm": float("nan"),
                "route_continuity_clf": float("nan"), "route_continuity_hmm": float("nan"),
                "route_precision": float("nan"), "route_recall": float("nan"),
                "route_f1": float("nan"), "n_fixes": int(len(track)), "n_valid_fixes": 0}
    px, py = net.proj.to_xy(track["lon"].to_numpy()[ok], track["lat"].to_numpy()[ok])
    labels = make_labels(px, py, net, trip.route)
    valid_fixes = pd.DataFrame({"fix_idx": idx, "t": track["t"].to_numpy()[ok],
                                "fix_valid": labels.fix_valid, "true_arc": labels.true_arc})
    clf_all = pred.fixes["arc_clf"].to_numpy(dtype="int64")[idx]
    hmm_all = pred.fixes["arc_hmm"].to_numpy(dtype="int64")[idx]
    metrics = compare_predictions(valid_fixes, clf_all, hmm_all, trip.route,
                                  net, art_gap(pred))
    # Compare predicted route against the reference route at route level.
    from .route_metrics import sequence_metrics
    seq = sequence_metrics(pred.route_arcs, trip.route, net)
    metrics.update({k: seq[k] for k in ("route_precision", "route_recall", "route_f1",
                                        "n_predicted_route_arcs", "n_reference_route_arcs")})
    metrics["n_fixes"] = int(len(track))
    metrics["n_valid_fixes"] = int(labels.fix_valid.sum())
    return metrics


def art_gap(pred: PredictionResult) -> float:
    return float(pred.diagnostics.get("gap_threshold_s", 60.0))
