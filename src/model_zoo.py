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

from dataclasses import dataclass, field
from typing import Callable

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


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
    # KKR: Gaussian Naive Bayes | Joshua: Random Forest, XGBoost | Julia: MLP  -> add entries here
}


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "(none yet)"
        raise KeyError(f"unknown model '{name}'. Registered: {known}")
    return REGISTRY[name]
