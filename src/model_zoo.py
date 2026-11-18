"""PathSnap model registry (Joshua's Random Forest and XGBoost models).

Factories accept keyword parameters because the shared ``src.evaluate.group_cv``
contract invokes them as ``factory(**params)``.  Each call returns a fresh
sklearn-compatible estimator that exposes ``predict_proba``.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import SEED

N_JOBS = max(1, min(8, os.cpu_count() or 1))

RF_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 22,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "max_samples": 0.8,
    "class_weight": None,
    "n_jobs": N_JOBS,
    "random_state": SEED,
}

XGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 350,
    "learning_rate": 0.06,
    "max_depth": 6,
    "min_child_weight": 3.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "scale_pos_weight": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbosity": 0,
}


def _pipeline(estimator) -> Pipeline:
    # This preprocessing is fitted within each CV training fold.  In particular,
    # no median is calculated from a validation or student-test fold.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", estimator),
    ])


def make_random_forest(**overrides: Any) -> Pipeline:
    """Random Forest (bagging ensemble) with probability output."""
    config = {**RF_DEFAULTS, **overrides}
    return _pipeline(RandomForestClassifier(**config))


def make_xgboost(**overrides: Any) -> Pipeline:
    """XGBoost gradient-boosted trees with binary probability output."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install XGBoost with `pip install xgboost`.") from exc
    config = {**XGB_DEFAULTS, **overrides}
    return _pipeline(XGBClassifier(**config))


# Tuple shape is fixed by TEAM.md: name -> (family, factory, config).
MODEL_ZOO: dict[str, tuple[str, Callable[..., Any], dict[str, Any]]] = {
    "random_forest": ("bagging ensemble", make_random_forest, RF_DEFAULTS.copy()),
    "xgboost": ("boosting ensemble", make_xgboost, XGB_DEFAULTS.copy()),
}
MODELS = MODEL_ZOO
MODEL_ALIASES = {
    "rf": "random_forest",
    "randomforest": "random_forest",
    "random_forest": "random_forest",
    "xgb": "xgboost",
    "xgboost": "xgboost",
}


def canonical_model_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = MODEL_ALIASES.get(key, key)
    if canonical not in MODEL_ZOO:
        raise KeyError(f"Unknown model {name!r}. Registered models: {', '.join(sorted(MODEL_ZOO))}")
    return canonical


def build_model(name: str, params: dict[str, Any] | None = None):
    """Construct a fresh estimator, preserving the shared factory(**params) API."""
    canonical = canonical_model_name(name)
    _, factory, _ = MODEL_ZOO[canonical]
    return factory(**(params or {}))


def register_model(name: str, family: str, factory: Callable[..., Any], config: dict[str, Any]) -> None:
    """Allow teammates to register their own model without overwriting entries."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        raise ValueError("Model name must not be empty")
    if key in MODEL_ZOO:
        raise ValueError(f"Model {key!r} is already registered")
    if not callable(factory):
        raise TypeError("factory must be callable")
    MODEL_ZOO[key] = (str(family), factory, dict(config))


def list_models() -> list[str]:
    return sorted(MODEL_ZOO)
