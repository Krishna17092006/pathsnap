import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src import evaluate as ev
from src.features import FEATURE_COLUMNS, load_or_build
from tests.synthetic import make_dataset


def _toy():
    """Two fixes in one trip, plus a labelled fix with no candidates (fix 2)."""
    cand = pd.DataFrame({
        "trip_id": ["a"] * 4, "fix_idx": [0, 0, 1, 1], "arc_id": [10, 11, 10, 12], "cand_rank": [0, 1, 0, 1],
        "label": [1, 0, 0, 1], "fix_valid": [True] * 4, "arc_u": [1, 2, 1, 5], "arc_v": [2, 1, 2, 6],
    })
    fixes = pd.DataFrame({"trip_id": ["a"] * 3, "fix_idx": [0, 1, 2], "fix_valid": [True] * 3,
                          "true_in_cands": [True, True, False]})
    return cand, fixes


def test_top1_counts_fixes_without_candidates_as_wrong():
    cand, fixes = _toy()
    m = ev.fix_metrics(cand, np.array([0.9, 0.1, 0.2, 0.8]), fixes)
    assert m["top1"] == pytest.approx(2 / 3)
    assert m["candidate_miss_rate"] == pytest.approx(1 / 3)
    assert m["n_valid_fixes"] == 3


def test_top1_tie_breaks_by_distance_rank():
    cand, fixes = _toy()
    m = ev.fix_metrics(cand, np.array([0.5, 0.5, 0.5, 0.5]), fixes)
    assert m["top1"] == pytest.approx(1 / 3)               # fix 0 right (rank 0), fix 1 wrong


def test_route_continuity():
    cand, _ = _toy()
    assert ev.route_continuity(cand, np.array([0.9, 0.1, 0.9, 0.1])) == 1.0   # arc 10 -> arc 10
    assert ev.route_continuity(cand, np.array([0.9, 0.1, 0.1, 0.9])) == 0.0   # arc 10 (ends 2) -> arc 12 (starts 5)


def test_candidate_metrics_perfect_and_confusion():
    m = ev.candidate_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.7])
    assert m["macro_f1"] == 1.0 and (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 0, 2)


def test_group_folds_are_trip_disjoint():
    cand = pd.DataFrame({"trip_id": [f"t{i}" for i in range(10) for _ in range(3)]})
    seen = set()
    for _, tr, va in ev.group_folds(cand, 5):
        assert not tr & va
        seen |= va
    assert seen == set(cand["trip_id"])


def test_training_rows_exclude_unlabelled_fixes():
    cand, _ = _toy()
    cand.loc[0, "fix_valid"] = False
    assert len(ev.training_rows(cand)) == 3


@pytest.fixture(scope="module")
def synthetic_train(tmp_path_factory):
    data_dir, processed_dir = make_dataset(tmp_path_factory.mktemp("cv"), n_train=6, n_test=1)
    cand = load_or_build("train", processed_dir, data_dir)
    fixes = pd.read_parquet(processed_dir / "fixes_train.parquet")
    return cand, fixes


def test_baseline_cv_runs_on_real_pipeline_tables(synthetic_train):
    cand, fixes = synthetic_train
    folds = ev.group_cv(None, {}, cand, fixes, FEATURE_COLUMNS, n_folds=3, verbose=False)
    assert len(folds) == 3
    assert folds["top1"].between(0, 1).all() and folds["log_loss"].isna().all()


def test_group_cv_with_placeholder_estimator(synthetic_train):
    """Pipeline mechanics only: a constant-probability placeholder, not one of the project's models."""
    cand, fixes = synthetic_train
    folds = ev.group_cv(DummyClassifier, {"strategy": "prior"}, cand, fixes, FEATURE_COLUMNS, n_folds=3,
                        verbose=False)
    expected = {"macro_f1", "top1", "log_loss", "route_continuity", "train_macro_f1", "train_time_s",
                "infer_ms_per_1k_fixes", "candidate_miss_rate"}
    assert expected <= set(folds.columns) and len(folds) == 3
    assert np.isfinite(folds["log_loss"]).all()


# ------------------------------------------------------------------ train.py / compare.py
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Runs train.main on synthetic trips with the baseline and a placeholder entry
    (DummyClassifier, not one of the project's models) registered only for this test."""
    from src import model_zoo, train
    root = tmp_path_factory.mktemp("e2e")
    data_dir, processed_dir = make_dataset(root, n_train=6, n_test=2)
    results = root / "results"
    model_zoo.REGISTRY["placeholder"] = model_zoo.ModelSpec("Test only", DummyClassifier,
                                                            {"params": {"strategy": "prior"}})
    try:
        train.main(["--model", "baseline", "--model", "placeholder", "--folds", "3", "--data-dir", str(data_dir),
                    "--processed-dir", str(processed_dir), "--results-dir", str(results)])
    finally:
        model_zoo.REGISTRY.pop("placeholder", None)
    return results


def test_train_writes_all_result_files(trained):
    import json
    for name in ["baseline", "placeholder"]:
        d = trained / name
        assert (d / "config.json").exists() and (d / "student_test_metrics.json").exists()
        assert len(pd.read_csv(d / "cv_folds.csv")) == 3
    table = pd.read_csv(trained / "model_comparison.csv")
    assert list(table["model"]) == ["baseline", "placeholder"]
    cfg = json.loads((trained / "placeholder" / "config.json").read_text())
    assert "label" not in cfg["features"] and "arc_id" not in cfg["features"]


def test_rerun_replaces_row_instead_of_duplicating(trained):
    from src import train
    path = trained / "model_comparison.csv"
    before = pd.read_csv(path)
    train.update_comparison([before.iloc[1].to_dict()], path)
    after = pd.read_csv(path)
    assert len(after) == len(before) and after["model"].is_unique


def test_unknown_model_fails_fast():
    from src import model_zoo
    with pytest.raises(KeyError):
        model_zoo.get_spec("does_not_exist")


def test_compare_makes_figures(trained):
    from src import compare
    paths = compare.make_all(trained)
    assert len(paths) == 5 and all(p.exists() and p.stat().st_size > 0 for p in paths)
