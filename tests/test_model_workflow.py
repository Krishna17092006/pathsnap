"""Integration tests for Joshua's models against PathSnap's shared evaluator."""
import numpy as np
import pandas as pd

from src.evaluate import evaluate_predictions, group_cv, group_folds
from src.features import FEATURE_COLUMNS
from src.model_selection import rank_models
from src.model_zoo import MODEL_ZOO, build_model, list_models
from src import tune, train as train_module
from src.tune import _cv_summary


def make_synthetic_tables(n_trips=10, fixes_per_trip=12, candidates_per_fix=3):
    rng = np.random.default_rng(42)
    candidate_rows, fix_rows = [], []
    for trip_no in range(n_trips):
        trip_id = f"{trip_no:08d}"
        for fix_idx in range(fixes_per_trip):
            positive = int(rng.integers(candidates_per_fix))
            for candidate_idx in range(candidates_per_fix):
                row = {
                    "trip_id": trip_id,
                    "fix_idx": fix_idx,
                    "arc_id": candidate_idx,
                    "cand_rank": candidate_idx,
                    "fix_valid": True,
                    "label": int(candidate_idx == positive),
                    "arc_u": candidate_idx,
                    "arc_v": candidate_idx + 1,
                }
                signal = float(candidate_idx == positive) + rng.normal(0, 0.2)
                for feature in FEATURE_COLUMNS:
                    row[feature] = signal if feature in {"dist_m", "cos_heading_diff"} else float(rng.normal())
                candidate_rows.append(row)
            fix_rows.append({
                "trip_id": trip_id,
                "fix_idx": fix_idx,
                "fix_valid": True,
                "true_in_cands": True,
            })
    return pd.DataFrame(candidate_rows), pd.DataFrame(fix_rows)


def test_factories_follow_shared_factory_kwargs_contract():
    assert set(("random_forest", "xgboost")).issubset(set(list_models()))
    for model_name in ("random_forest", "xgboost"):
        estimator = build_model(model_name, {"n_estimators": 12})
        x = np.random.default_rng(42).normal(size=(80, len(FEATURE_COLUMNS)))
        y = np.array([0, 1] * 40)
        estimator.fit(x, y)
        p = estimator.predict_proba(x[:7])[:, 1]
        assert p.shape == (7,)
        assert np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()


def test_shared_group_cv_is_trip_disjoint_and_scores_tree_model():
    candidates, fixes = make_synthetic_tables()
    for _, train_ids, valid_ids in group_folds(candidates, n_folds=5):
        assert train_ids.isdisjoint(valid_ids)

    family, factory, defaults = MODEL_ZOO["random_forest"]
    folds = group_cv(
        factory=factory,
        params={**defaults, "n_estimators": 15, "max_depth": 8},
        cand=candidates,
        fixes=fixes,
        features=FEATURE_COLUMNS,
        n_folds=5,
        max_train_rows=500,
        verbose=False,
    )
    assert len(folds) == 5
    assert folds["top1"].between(0, 1).all()
    assert folds["macro_f1"].between(0, 1).all()
    assert (folds["log_loss"] >= 0).all()


def test_tuner_summary_uses_shared_evaluate_group_cv():
    candidates, fixes = make_synthetic_tables()
    summary, folds = _cv_summary(
        "random_forest", candidates, fixes,
        {"n_estimators": 12, "max_depth": 8}, max_train_rows=500,
    )
    assert len(folds) == 5
    assert 0 <= summary["top1_mean"] <= 1
    assert 0 <= summary["macro_f1_mean"] <= 1


def test_fix_miss_is_counted_wrong_by_shared_evaluator():
    candidates = pd.DataFrame([
        {"trip_id": "a", "fix_idx": 0, "arc_id": 0, "cand_rank": 0, "label": 1, "fix_valid": True},
        {"trip_id": "a", "fix_idx": 0, "arc_id": 1, "cand_rank": 1, "label": 0, "fix_valid": True},
        {"trip_id": "a", "fix_idx": 1, "arc_id": 0, "cand_rank": 0, "label": 0, "fix_valid": True},
    ])
    fixes = pd.DataFrame([
        {"trip_id": "a", "fix_idx": 0, "fix_valid": True, "true_in_cands": True},
        {"trip_id": "a", "fix_idx": 1, "fix_valid": True, "true_in_cands": False},
    ])
    metrics = evaluate_predictions(candidates, np.array([0.8, 0.2, 0.9]), fixes)
    assert metrics["top1"] == 0.5
    assert metrics["candidate_miss_rate"] == 0.5


def test_model_selection_uses_cv_metrics_not_student_test():
    frame = pd.DataFrame([
        {"model": "random_forest", "cv_top1_mean": 0.82, "cv_macro_f1_mean": 0.71,
         "cv_log_loss_mean": 0.42, "student_test_top1": 0.90, "inference_ms_per_1000_fixes": 30},
        {"model": "xgboost", "cv_top1_mean": 0.84, "cv_macro_f1_mean": 0.70,
         "cv_log_loss_mean": 0.48, "student_test_top1": 0.60, "inference_ms_per_1000_fixes": 45},
    ])
    ranked, primary, _ = rank_models(frame)
    assert primary == "cv_top1"
    assert ranked.iloc[0]["model"] == "xgboost"


def test_one_trial_optuna_workflow_writes_expected_artifacts(tmp_path, monkeypatch):
    candidates, fixes = make_synthetic_tables(n_trips=10, fixes_per_trip=5, candidates_per_fix=3)
    # Keep this test deliberately tiny; it checks the complete workflow plumbing,
    # not model quality or the runtime of the production search space.
    monkeypatch.setattr(
        tune, "_load_feature_tables",
        lambda force=False: (candidates, fixes, candidates.copy(), fixes.copy()),
    )
    monkeypatch.setattr(tune, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(tune, "ARTIFACTS_DIR", tmp_path / "artifacts")
    original = tune.MODEL_ZOO["random_forest"]
    monkeypatch.setitem(
        tune.MODEL_ZOO, "random_forest",
        (original[0], original[1], {"n_estimators": 3, "max_depth": 5, "n_jobs": 1, "random_state": 42}),
    )

    summary = tune.run_tuning(
        "random_forest", n_trials=1, objective_metric="top1",
        force_features=False, max_train_rows=100,
    )
    out = tmp_path / "results" / "tuning" / "random_forest"
    assert summary["model"] == "random_forest"
    assert (out / "optuna_trials.csv").exists()
    assert (out / "tuned_vs_untuned.csv").exists()
    assert (out / "tuned_model.joblib").exists()
    assert (out / "best_params.json").exists()


def test_train_entrypoint_writes_cv_holdout_and_model_files(tmp_path, monkeypatch):
    train_candidates, train_fixes = make_synthetic_tables(n_trips=10, fixes_per_trip=5)
    test_candidates, test_fixes = make_synthetic_tables(n_trips=5, fixes_per_trip=5)
    test_candidates["trip_id"] = test_candidates["trip_id"].map(lambda value: "test_" + value)
    test_fixes["trip_id"] = test_fixes["trip_id"].map(lambda value: "test_" + value)

    processed = tmp_path / "processed"
    processed.mkdir()
    train_fixes_path = processed / "fixes_train.parquet"
    student_fixes_path = processed / "fixes_student_test.parquet"
    train_fixes_path.touch(); student_fixes_path.touch()

    monkeypatch.setattr(train_module, "PROCESSED_DIR", processed)
    monkeypatch.setattr(train_module, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(train_module, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        train_module, "load_or_build",
        lambda partition, rebuild=False: train_candidates if partition == "train" else test_candidates,
    )
    monkeypatch.setattr(
        pd, "read_parquet",
        lambda path: train_fixes if "fixes_train" in str(path) else test_fixes,
    )
    original = train_module.MODEL_ZOO["random_forest"]
    monkeypatch.setitem(
        train_module.MODEL_ZOO, "random_forest",
        (original[0], original[1], {"n_estimators": 8, "max_depth": 5, "n_jobs": 1, "random_state": 42}),
    )

    result = train_module.train_one("random_forest", rebuild_features=False, n_folds=5, max_train_rows=100)
    out = tmp_path / "results" / "models" / "random_forest"
    assert result["model"] == "random_forest"
    assert (out / "cv_folds.csv").exists()
    assert (out / "student_test_metrics.json").exists()
    assert (out / "model.joblib").exists()
    assert (tmp_path / "results" / "models" / "model_comparison.csv").exists()
