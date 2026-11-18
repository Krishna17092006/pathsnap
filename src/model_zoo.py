"""Model registry: name -> ModelSpec(family, factory, config).

Each member adds their own models here (TEAM.md "Model ownership"). A factory
takes the `params` part of the config as keyword arguments and returns an
unfitted sklearn estimator or Pipeline with `predict_proba`. Put scaling inside
the Pipeline, so it is fitted on the training fold only.

Config keys
    params          keyword arguments for the factory
    max_train_rows  optional cap on training rows (stratified sample), for slow models
    description     short "key config" text for the comparison table

Example entry:

    from sklearn.ensemble import RandomForestClassifier

    def make_rf(n_estimators=300, max_depth=None):
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      n_jobs=-1, random_state=SEED)

    REGISTRY["random_forest"] = ModelSpec("Bagging ensemble", make_rf, {
        "params": {"n_estimators": 300},
        "description": "RandomForest, 300 trees",
    })

Then run `python -m src.train --model random_forest`.
The nearest-arc baseline is not a model; train.py handles the name "baseline".
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .naive_bayes import gnb_decorrelated


@dataclass(frozen=True)
class ModelSpec:
    family: str
    factory: Callable
    config: dict = field(default_factory=dict)

    @property
    def params(self) -> dict:
        return dict(self.config.get("params", {}))

    @property
    def max_train_rows(self):
        return self.config.get("max_train_rows")

    @property
    def description(self) -> str:
        return self.config.get("description", "")


# ------------------------------------------------------------------ Krishna Jha: models 1-2
def make_logreg(C: float = 1.0, class_weight=None, max_iter: int = 2000) -> Pipeline:
    """Model 1. Linear decision boundary on standardised features; well-calibrated probabilities
    when unweighted, which matters because the HMM uses them as emissions."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter, random_state=SEED)),
    ])


def make_knn(n_neighbors: int = 50, weights: str = "uniform", algorithm: str = "auto") -> Pipeline:
    """Model 2. Instance-based: a candidate is scored by the labels of its most similar training
    candidates. Scaling is essential, otherwise metre-valued features dominate the distance."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=-1)),
    ])


# ------------------------------------------------------------------ Joshua: models 4-5
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


def _tree_pipeline(estimator) -> Pipeline:
    # The imputer is fitted within each CV training fold, so no median comes from a validation fold.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", estimator),
    ])


def make_random_forest(**overrides: Any) -> Pipeline:
    """Model 4. Random Forest (bagging ensemble) with probability output."""
    return _tree_pipeline(RandomForestClassifier(**{**RF_DEFAULTS, **overrides}))


def make_xgboost(**overrides: Any) -> Pipeline:
    """Model 5. XGBoost gradient-boosted trees with binary probability output."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install XGBoost with `pip install xgboost` (macOS also needs `brew install libomp`).") from exc
    return _tree_pipeline(XGBClassifier(**{**XGB_DEFAULTS, **overrides}))


REGISTRY: dict[str, ModelSpec] = {
    "logreg": ModelSpec("Linear / probabilistic", make_logreg, {
        "params": {"C": 1.0, "class_weight": None},
        "description": "StandardScaler + LogisticRegression, C=1, unweighted",
    }),
    "logreg_balanced": ModelSpec("Linear / probabilistic", make_logreg, {
        "params": {"C": 1.0, "class_weight": "balanced"},
        "description": "StandardScaler + LogisticRegression, C=1, class_weight=balanced",
    }),
    "knn": ModelSpec("Instance-based", make_knn, {
        "params": {"n_neighbors": 50, "weights": "uniform"},
        "max_train_rows": 300_000,
        "description": "StandardScaler + KNN, k=50, uniform weights, 300k-row stratified training sample",
    }),
    # KKR: model 3. The configuration selected by mean CV Top-1 in src/naive_bayes.py, which compares
    # three Naive Bayes configurations (results/models/naive_bayes/). Registered as "gaussian_nb" so
    # train.py's outputs do not overwrite that study.
    "gaussian_nb": ModelSpec("Probabilistic", gnb_decorrelated, {
        "params": {"var_smoothing": 1e-9, "corr_threshold": 0.8},
        "description": "QuantileTransformer(normal) + correlation filter |r|>0.8 (fitted per fold) + GaussianNB",
    }),
    "random_forest": ModelSpec("Bagging ensemble", make_random_forest, {
        "params": RF_DEFAULTS.copy(),
        "description": "RandomForest, 300 trees, max_depth=22, min_samples_leaf=2, max_features=sqrt, max_samples=0.8",
    }),
    "xgboost": ModelSpec("Boosting ensemble", make_xgboost, {
        "params": XGB_DEFAULTS.copy(),
        "description": "XGBoost hist, 350 trees, lr=0.06, max_depth=6, subsample/colsample 0.85, reg_lambda=2",
    }),
    # Julia: MLP -> add an entry here
}


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "(none yet)"
        raise KeyError(f"unknown model '{name}'. Registered: {known}")
    return REGISTRY[name]


# ------------------------------------------------------------------ tuple view used by src/tune.py
# name -> (family, factory, default params). Kept in sync with REGISTRY.
MODEL_ZOO: dict[str, tuple[str, Callable[..., Any], dict[str, Any]]] = {
    name: (spec.family, spec.factory, spec.params) for name, spec in REGISTRY.items()
}
MODELS = MODEL_ZOO
MODEL_ALIASES = {"rf": "random_forest", "randomforest": "random_forest", "xgb": "xgboost"}


def canonical_model_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = MODEL_ALIASES.get(key, key)
    if canonical not in MODEL_ZOO:
        raise KeyError(f"Unknown model {name!r}. Registered models: {', '.join(sorted(MODEL_ZOO))}")
    return canonical


def build_model(name: str, params: dict[str, Any] | None = None):
    """Construct a fresh estimator with the shared factory(**params) contract."""
    _, factory, _ = MODEL_ZOO[canonical_model_name(name)]
    return factory(**(params or {}))


def register_model(name: str, family: str, factory: Callable[..., Any], config: dict[str, Any]) -> None:
    """Register a model at runtime without overwriting an existing entry."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        raise ValueError("Model name must not be empty")
    if key in REGISTRY:
        raise ValueError(f"Model {key!r} is already registered")
    if not callable(factory):
        raise TypeError("factory must be callable")
    REGISTRY[key] = ModelSpec(str(family), factory, {"params": dict(config)})
    MODEL_ZOO[key] = (str(family), factory, dict(config))


def list_models() -> list[str]:
    return sorted(REGISTRY)
