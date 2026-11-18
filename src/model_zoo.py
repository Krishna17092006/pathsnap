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

from .config import SEED  # noqa: F401  (for model factories: random_state=SEED)


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


REGISTRY: dict[str, ModelSpec] = {
    # Krishna Jha: Logistic Regression, KNN | KKR: Gaussian Naive Bayes
    # Joshua: Random Forest, XGBoost        | Julia: MLP
}


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "(none yet)"
        raise KeyError(f"unknown model '{name}'. Registered: {known}")
    return REGISTRY[name]
