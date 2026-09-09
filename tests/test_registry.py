from __future__ import annotations

import sys
from pathlib import Path

# Add packages/ml_engine to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))

from models.registry import (
    get_estimator,
    list_models,
    is_regression_model,
    supports_proba,
)


def test_list_models_classification_and_regression():
    cls_models = list_models("classification")
    reg_models = list_models("regression")

    assert "logistic_regression" in cls_models
    assert "random_forest" in cls_models

    assert "linear_regression" in reg_models
    assert "random_forest_regressor" in reg_models


def test_get_estimator_regression_models():
    reg_rf = get_estimator("random_forest_regressor", random_seed=123)
    assert reg_rf.random_state == 123
    assert is_regression_model("random_forest_regressor")
    assert not supports_proba("random_forest_regressor")

    lin_reg = get_estimator("linear_regression")
    assert lin_reg is not None
    assert is_regression_model("linear_regression")
