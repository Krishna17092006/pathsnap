"""Every registered classifier must work as the HMM emission model in src.pipeline."""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.model_zoo import REGISTRY
from src.pipeline import _model_probabilities, model_input


def _rows(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (X["dist_m"] < -0.3).astype(int).to_numpy()
    return X, y


def _art(model):
    return {"model_bundle": model, "feature_columns": list(FEATURE_COLUMNS)}


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_registered_model_gives_valid_emission_probabilities(name):
    spec = REGISTRY[name]
    params = spec.params
    if "n_estimators" in params:
        params["n_estimators"] = 10
    if name == "knn":
        params["n_neighbors"] = 5
    X, y = _rows()
    model = spec.factory(**params).fit(X.to_numpy(), y)
    p = _model_probabilities(X.iloc[:50], _art(model))
    assert p.shape == (50,)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(p, model.predict_proba(X.iloc[:50].to_numpy())[:, 1])


def test_model_fitted_on_dataframe_is_matched_by_column_name_without_warning():
    # src.tune fits on a DataFrame; the pipeline must not pass a bare array or reorder columns.
    X, y = _rows()
    model = REGISTRY["random_forest"].factory(n_estimators=10).fit(X, y)
    shuffled = X.iloc[:50, ::-1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = _model_probabilities(shuffled, _art(model))
    np.testing.assert_allclose(p, model.predict_proba(X.iloc[:50])[:, 1])


def test_imputing_pipeline_receives_nan_but_plain_model_gets_zero():
    X, y = _rows()
    tree = REGISTRY["random_forest"].factory(n_estimators=10).fit(X.to_numpy(), y)
    linear = REGISTRY["logreg"].factory().fit(X.to_numpy(), y)
    test = X.iloc[:5].copy()
    test.iloc[0, 0] = np.nan
    assert np.isnan(model_input(tree, test)[0, 0])        # imputer applies training median
    assert model_input(linear, test)[0, 0] == 0.0         # no imputer: must not see NaN
    assert np.all(np.isfinite(_model_probabilities(test, _art(linear))))
