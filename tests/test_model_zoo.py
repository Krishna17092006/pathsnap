import numpy as np
import pytest

from src.model_zoo import REGISTRY, get_spec


@pytest.mark.parametrize("name", ["logreg", "logreg_balanced", "knn"])
def test_registered_models_fit_and_predict_probabilities(name):
    spec = get_spec(name)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 19))
    y = (X[:, 0] + 0.3 * rng.normal(size=300) > 0).astype(int)
    model = spec.factory(**spec.params).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (300,) and np.all((p >= 0) & (p <= 1))
    assert spec.family and spec.description


def test_registry_names_are_unique_and_known():
    assert {"logreg", "logreg_balanced", "knn"} <= set(REGISTRY)
