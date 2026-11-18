import numpy as np
import pandas as pd
import pytest

from src.data_io import Trip
from src.geo import LocalProjection
from src.hmm import DecoderConfig, viterbi_decode
from src.network import RoadNetwork
from src.pipeline import PredictionResult, predict_trip, score_trip
from src.route_metrics import compare_predictions, route_continuity, sequence_metrics


def _network(arcs=None):
    nodes = pd.DataFrame({
        "lon": [0.0, 0.0001, 0.0002, 0.0010, 0.0011],
        "lat": [0.0] * 5,
    })
    arcs = pd.DataFrame(arcs or [(0, 1), (1, 2), (3, 4)], columns=["u", "v"])
    proj = LocalProjection.from_points(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    return RoadNetwork(nodes, arcs, proj)


def _fixes(times=(0.0, 1.0), lons=(0.00002, 0.00018)):
    return pd.DataFrame({"fix_idx": np.arange(len(times)), "t": times,
                         "lon": lons, "lat": np.zeros(len(times))})


def test_viterbi_prefers_connected_chain_over_higher_local_probability():
    net = _network()
    fixes = _fixes()
    cand = pd.DataFrame({
        "fix_idx": [0, 0, 1, 1], "arc_id": [0, 2, 1, 2],
        "dist_m": [1.0, 0.5, 1.0, 0.5], "t_along": [0.2, 0.5, 0.8, 0.5],
    })
    # The classifier strongly favours the disconnected arc at fix 1.
    prob = np.array([0.9, 0.1, 0.01, 0.99])
    decoded = viterbi_decode(cand, fixes, net, prob,
                             DecoderConfig(beta=3, gap_threshold_s=60))
    assert decoded.tolist() == [0, 1]


def test_missing_candidate_fix_is_unmatched_and_chain_bridges_it():
    net = _network()
    fixes = pd.DataFrame({"fix_idx": [0, 1, 2], "t": [0.0, 1.0, 2.0],
                          "lon": [0.00002, 0.0001, 0.00018], "lat": [0.0] * 3})
    cand = pd.DataFrame({"fix_idx": [0, 2], "arc_id": [0, 1],
                         "dist_m": [1.0, 1.0], "t_along": [0.2, 0.8]})
    decoded = viterbi_decode(cand, fixes, net, config=DecoderConfig())
    assert decoded.tolist() == [0, -1, 1]


def test_disconnected_transition_restarts_chain_and_large_gap_splits():
    net = _network()
    fixes = pd.DataFrame({"fix_idx": [0, 1], "t": [0.0, 1.0],
                          "lon": [0.00002, 0.00108], "lat": [0.0, 0.0]})
    cand = pd.DataFrame({"fix_idx": [0, 1], "arc_id": [0, 2],
                         "dist_m": [0.1, 0.1], "t_along": [0.2, 0.8]})
    # A disconnected transition is impossible; Viterbi restarts at fix 1.
    assert viterbi_decode(cand, fixes, net).tolist() == [0, 2]
    # A large time gap also restarts, even if the route graph is connected.
    fixes.loc[1, "t"] = 100.0
    assert viterbi_decode(cand, fixes, net).tolist() == [0, 2]


def test_duplicate_directed_arcs_are_treated_as_same_segment():
    net = _network(arcs=[(0, 1), (0, 1), (1, 2), (3, 4)])
    fixes = _fixes()
    cand = pd.DataFrame({"fix_idx": [0, 1], "arc_id": [0, 1],
                         "dist_m": [1.0, 1.0], "t_along": [0.2, 0.8]})
    assert viterbi_decode(cand, fixes, net).tolist() == [0, 1]


def test_route_metrics_use_temporal_connectivity_and_reference_route():
    net = _network()
    fixes = pd.DataFrame({"fix_idx": [0, 1, 2], "t": [0.0, 1.0, 2.0]})
    assert route_continuity(fixes, [0, 1, 2], net) == pytest.approx(1 / 2)
    m = sequence_metrics([0, 2], [0, 1], net)
    assert m["route_precision"] == pytest.approx(0.5)
    assert m["route_recall"] == pytest.approx(0.5)


def test_predict_trip_does_not_require_reference_route():
    net = _network()
    track = pd.DataFrame({"lon": [0.00002, 0.00008, 0.00014],
                          "lat": [0.0, 0.0, 0.0], "t": [0.0, 1.0, 2.0]})
    trip = Trip("synthetic", "student_test", track,
                pd.DataFrame({"lon": net.node_lon, "lat": net.node_lat}),
                pd.DataFrame({"u": net.u, "v": net.v}), None)
    art = {"model_bundle": None, "feature_columns": [],
           "decoder_config": DecoderConfig(), "artifact_dir": ""}
    pred = predict_trip(trip, art)
    assert {"fix_idx", "t", "lon", "lat", "arc_clf", "prob_clf", "arc_hmm"} <= set(pred.fixes)
    assert len(pred.fixes) == len(track)
    assert pred.fixes.arc_hmm.ge(-1).all()
    metrics = score_trip(pred, trip)
    assert metrics["n_valid_fixes"] == 0


def test_comparison_scores_hmm_against_reference_not_itself():
    net = _network()
    fixes = pd.DataFrame({"fix_idx": [0, 1], "t": [0.0, 1.0],
                          "fix_valid": [True, True], "true_arc": [0, 1]})
    metrics = compare_predictions(fixes, [0, 2], [0, 2], [0, 1], net)
    assert metrics["hmm_route_precision"] == pytest.approx(0.5)
    assert metrics["hmm_route_recall"] == pytest.approx(0.5)


def test_score_trip_reports_route_level_metrics_against_trip_route():
    net = _network()
    track = pd.DataFrame({"lon": [0.00002, 0.00008], "lat": [0.0, 0.0], "t": [0.0, 1.0]})
    trip = Trip("synthetic", "student_test", track,
                pd.DataFrame({"lon": net.node_lon, "lat": net.node_lat}),
                pd.DataFrame({"u": net.u, "v": net.v}), np.array([0, 1]))
    fixes = pd.DataFrame({"fix_idx": [0, 1], "t": [0.0, 1.0],
                          "lon": track.lon, "lat": track.lat,
                          "arc_clf": [0, 2], "prob_clf": [0.9, 0.9], "arc_hmm": [0, 2]})
    pred = PredictionResult(fixes, [0, 2], net.route_polylines([0, 2]), {}, net)
    metrics = score_trip(pred, trip)
    assert metrics["route_precision"] == pytest.approx(0.5)
    assert metrics["route_recall"] == pytest.approx(0.5)
    assert metrics["hmm_route_precision"] == pytest.approx(0.5)
