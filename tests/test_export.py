import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import export as ex
from src.features import FEATURE_COLUMNS
from src.model_zoo import get_spec
from tests.synthetic import make_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return make_dataset(tmp_path_factory.mktemp("export"), n_train=5, n_test=1)


def _export(tmp_path, dataset, **kw):
    data_dir, processed_dir = dataset
    return ex.export(model=kw.pop("model", "logreg"), out_dir=tmp_path / "artifacts",
                     results_dir=kw.pop("results_dir", tmp_path / "results"),
                     processed_dir=processed_dir, data_dir=data_dir, **kw)


def test_refit_writes_all_four_files_and_they_load(tmp_path, dataset):
    out = _export(tmp_path, dataset, source="refit")
    assert sorted(p.name for p in out.iterdir()) == sorted(["model.joblib", "features.json", "hmm.json", "metadata.json"])
    art = ex.read_artifacts(out)
    assert art.features == FEATURE_COLUMNS
    assert art.metadata["model"] == "logreg" and art.metadata["source"] == "refit"
    assert art.metadata["details"]["train_trips"] == 5
    assert art.hmm["status"].startswith("default")
    p = art.predict_proba(pd.DataFrame(np.zeros((3, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS))
    assert p.shape == (3,)


def test_auto_prefers_the_tuned_model(tmp_path, dataset):
    results = tmp_path / "results"
    tuned_dir = results / "tuning" / "logreg"
    tuned_dir.mkdir(parents=True)
    spec = get_spec("logreg")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURE_COLUMNS)))
    model = spec.factory(C=0.5).fit(X, (X[:, 0] > 0).astype(int))
    joblib.dump(model, tuned_dir / "tuned_model.joblib")
    (tuned_dir / "best_params.json").write_text(json.dumps({"best_params": {"C": 0.5}, "objective": "group_cv_top1",
                                                            "best_value": 0.9, "n_trials_total": 3}))
    pd.DataFrame([{"configuration": "untuned", "cv_top1_mean": 0.8},
                  {"configuration": "tuned", "cv_top1_mean": 0.9}]).to_csv(tuned_dir / "tuned_vs_untuned.csv", index=False)
    out = _export(tmp_path, dataset, source="auto", results_dir=results)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["source"] == "tuned" and meta["params"] == {"C": 0.5}
    assert meta["details"]["tuning"]["n_trials_total"] == 3
    assert len(meta["details"]["tuned_vs_untuned"]) == 2


def test_source_tuned_without_tuning_results_fails_clearly(tmp_path, dataset):
    with pytest.raises(FileNotFoundError, match="src.tune"):
        _export(tmp_path, dataset, source="tuned")


def test_hmm_config_is_used(tmp_path, dataset):
    cfg = tmp_path / "decoder.json"
    cfg.write_text(json.dumps({"beta_m": 3.2, "gap_split_s": 45}))
    out = _export(tmp_path, dataset, source="refit", hmm_config=cfg)
    hmm = json.loads((out / "hmm.json").read_text())
    assert hmm["beta_m"] == 3.2 and hmm["status"].startswith("tuned")


def test_check_detects_tampering(tmp_path, dataset):
    out = _export(tmp_path, dataset, source="refit")
    (out / "features.json").write_text(json.dumps(list(reversed(FEATURE_COLUMNS))))
    with pytest.raises(ValueError, match="features.json"):
        ex.check_artifacts(out)
    out = _export(tmp_path, dataset, source="refit")
    with open(out / "model.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        ex.read_artifacts(out)


def test_missing_feature_columns_rejected(tmp_path, dataset):
    art = ex.read_artifacts(_export(tmp_path, dataset, source="refit"))
    with pytest.raises(ValueError, match="missing feature columns"):
        art.predict_proba(pd.DataFrame({"dist_m": [1.0]}))
