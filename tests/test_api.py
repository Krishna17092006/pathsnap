"""Integration tests for api/main.py with the real artifacts and a small real student_test trip."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import src.pipeline as pipeline

ROOT = Path(__file__).resolve().parents[1]
TRIPS = ROOT / "data" / "student_test"
SMALL_TRIP = "00000052"            # 262 fixes: the smallest student_test trip
pytestmark = pytest.mark.skipif(not (TRIPS / SMALL_TRIP).is_dir() or not (ROOT / "artifacts" / "model.joblib").exists(),
                                reason="needs data/student_test and artifacts/")


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setenv("PATHSNAP_TRIPS_DIR", "data/student_test")
    mp.delenv("PATHSNAP_ARTIFACTS_DIR", raising=False)
    calls = {"load_artifacts": 0, "predict_trip": 0}
    real_load, real_predict = api_main.load_artifacts, api_main.predict_trip

    def counting_load(*a, **k):
        calls["load_artifacts"] += 1
        return real_load(*a, **k)

    def counting_predict(*a, **k):
        calls["predict_trip"] += 1
        return real_predict(*a, **k)

    mp.setattr(api_main, "load_artifacts", counting_load)
    mp.setattr(api_main, "predict_trip", counting_predict)
    with TestClient(api_main.app) as c:
        c.calls = calls
        yield c
    mp.undo()


def test_health_is_real(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["mode"] == "real" and body["model"] == "xgboost"
    assert "mock" not in body


def test_trips_lists_real_trip_ids(client):
    trips = client.get("/trips").json()["trips"]
    assert SMALL_TRIP in trips and trips == sorted(p.name for p in TRIPS.iterdir() if p.is_dir())


@pytest.mark.parametrize("bad", ["99999999", "../../etc", "abc", "0000005"])
def test_unknown_or_invalid_trip_ids_return_404(client, bad):
    assert client.post(f"/predict/{bad}").status_code == 404


def test_predict_uses_real_pipeline_and_returns_real_json(client):
    before = client.calls["predict_trip"]
    r = client.post(f"/predict/{SMALL_TRIP}")
    assert r.status_code == 200
    assert client.calls["predict_trip"] == before + 1            # the real predict_trip was called
    body = r.json()
    json.dumps(body, allow_nan=False)                             # strictly JSON serialisable, no NaN
    assert body["mode"] == "real" and "mock" not in body
    assert body["trip_id"] == SMALL_TRIP and body["n_fixes"] == 262 and len(body["gps"]) == 262
    assert len(body["predicted_arcs"]) > 0 and all(isinstance(a, int) for a in body["predicted_arcs"])
    assert body["predicted_route"] and all(len(pt) == 2 for line in body["predicted_route"] for pt in line)
    assert body["timing_ms"]["total"] > 0 and body["timing_ms"]["hmm"] is not None
    assert body["metrics"]["n_candidates"] > 0


def test_predict_never_returns_reference_data(client):
    body = client.post(f"/predict/{SMALL_TRIP}").json()
    text = json.dumps(body)
    for forbidden in ("true_arc", "fix_valid", "reference", "top1", "route_precision", "route_recall", "label"):
        assert forbidden not in text
    assert set(body) == {"trip_id", "n_fixes", "gps", "predicted_route", "predicted_arcs", "metrics",
                         "timing_ms", "mode"}


def test_predict_does_not_open_the_route_file(client, monkeypatch):
    opened = []
    real_read_csv = pd.read_csv

    def spy(path, *a, **k):
        opened.append(str(path))
        return real_read_csv(path, *a, **k)

    monkeypatch.setattr(pd, "read_csv", spy)
    assert client.post(f"/predict/{SMALL_TRIP}").status_code == 200
    assert opened and not any(p.endswith(".route") for p in opened)


def test_predict_trip_receives_no_route(client, monkeypatch):
    seen = []
    real = pipeline.predict_trip

    def spy(trip, art, *a, **k):
        seen.append(trip.route)
        return real(trip, art, *a, **k)

    monkeypatch.setattr(api_main, "predict_trip", spy)
    client.post(f"/predict/{SMALL_TRIP}")
    assert seen == [None]


def test_artifacts_loaded_once_not_per_request(client):
    for _ in range(2):
        assert client.post(f"/predict/{SMALL_TRIP}").status_code == 200
    assert client.calls["load_artifacts"] == 1


def test_evaluate_returns_only_aggregate_metrics(client):
    r = client.post(f"/evaluate/{SMALL_TRIP}")
    assert r.status_code == 200
    body = r.json()
    json.dumps(body, allow_nan=False)
    assert set(body) == {"trip_id", "metrics", "timing_ms", "mode"}
    m = body["metrics"]
    assert 0 <= m["top1_classifier"] <= 1 and 0 <= m["top1_hmm"] <= 1
    assert all(not isinstance(v, (list, dict)) for v in m.values())      # no per-fix data or routes


def test_malformed_trip_returns_422(client, tmp_path, monkeypatch):
    bad = tmp_path / "student_test" / "12345678"
    bad.mkdir(parents=True)
    (bad / "12345678.track").write_text("1 2 3\n")                       # no road network files
    monkeypatch.setattr(api_main.app.state, "trips_dir", tmp_path / "student_test")
    r = client.post("/predict/12345678")
    assert r.status_code == 422 and "malformed" in r.json()["detail"]


def test_startup_fails_without_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("PATHSNAP_ARTIFACTS_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="artifacts missing"):
        with TestClient(api_main.app):
            pass
