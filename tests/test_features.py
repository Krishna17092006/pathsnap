import numpy as np
import pandas as pd
import pytest

from src.data_io import load_trip
from src.features import FEATURE_COLUMNS, FORBIDDEN_FEATURES, META_COLUMNS, add_features, fix_kinematics
from src.geo import LocalProjection
from src.network import RoadNetwork
from tests.synthetic import make_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("syn"), n_train=2, n_test=1)


def _trip_and_net(data_dir, tid="00000001", part="train"):
    trip = load_trip(tid, part, data_dir)
    proj = LocalProjection.from_points(trip.track["lon"], trip.track["lat"])
    return trip, RoadNetwork(trip.nodes, trip.arcs, proj)


def _cand(processed_dir, tid="00000001", part="train"):
    c = pd.read_parquet(processed_dir / f"candidates_{part}.parquet")
    return c[c["trip_id"] == tid].reset_index(drop=True)


def test_no_label_or_id_columns_are_features():
    assert not FORBIDDEN_FEATURES & set(FEATURE_COLUMNS)


def test_features_complete_and_finite(dataset):
    data_dir, processed_dir = dataset
    trip, net = _trip_and_net(data_dir)
    cand = _cand(processed_dir)
    out = add_features(cand, trip, net)
    assert len(out) == len(cand)
    assert set(cand.columns) <= set(out.columns)
    assert set(FEATURE_COLUMNS + META_COLUMNS) <= set(out.columns)
    assert np.isfinite(out[FEATURE_COLUMNS].to_numpy(dtype=float)).all()
    assert out["heading_diff_deg"].between(0, 180).all()
    assert out["cos_heading_diff"].between(-1, 1).all()


def test_features_do_not_depend_on_route_or_labels(dataset):
    data_dir, processed_dir = dataset
    trip, net = _trip_and_net(data_dir)
    cand = _cand(processed_dir)
    ref = add_features(cand, trip, net)[FEATURE_COLUMNS]
    trip.route = None                                     # inference: no reference route
    scrambled = cand.copy()
    scrambled["label"] = np.random.default_rng(0).integers(0, 2, len(cand))
    scrambled["true_arc"] = -1
    pd.testing.assert_frame_equal(add_features(scrambled, trip, net)[FEATURE_COLUMNS], ref)


def test_heading_separates_two_directions(dataset):
    """On the true arc the vehicle heading should agree with the arc direction; on its reverse it should not."""
    data_dir, processed_dir = dataset
    trip, net = _trip_and_net(data_dir)
    out = add_features(_cand(processed_dir), trip, net)
    moving = (out["has_heading"] == 1) & out["fix_valid"]
    true_cos = out.loc[moving & (out["label"] == 1), "cos_heading_diff"]
    rev = moving & (out["label"] == 0) & (out["dist_minus_min"] < 0.01) & (out["is_two_way"] == 1)
    assert true_cos.median() > 0.8
    if rev.any():
        assert out.loc[rev, "cos_heading_diff"].median() < true_cos.median()


def test_kinematics_handles_gaps_and_stops():
    track = pd.DataFrame({"lon": [10.0, 10.0, 10.0001, 10.0002, 10.0002],
                          "lat": [50.0, 50.0, 50.0, 50.0, 50.0], "t": [0, 1, 2, 100, 101]})
    kin = fix_kinematics(track, LocalProjection(10.0, 50.0))
    assert np.isnan(kin["heading_deg"].iloc[0])           # did not move: no heading
    assert np.isnan(kin["heading_deg"].iloc[2])           # spans a 98 s gap: no heading
    assert (kin["speed_mps"] >= 0).all() and np.isfinite(kin[["log_dt_prev", "log_dt_next"]]).all().all()
    assert kin["log_dt_next"].iloc[2] == pytest.approx(np.log1p(98))


def test_rejects_bad_input(dataset):
    data_dir, processed_dir = dataset
    trip, net = _trip_and_net(data_dir)
    cand = _cand(processed_dir)
    with pytest.raises(ValueError):
        add_features(cand.drop(columns=["dist_m"]), trip, net)
    bad = cand.copy()
    bad.loc[0, "arc_id"] = net.n_arcs + 5
    with pytest.raises(ValueError):
        add_features(bad, trip, net)
