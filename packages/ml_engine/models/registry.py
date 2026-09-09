from __future__ import annotations

"""
Model registry for MVP classifiers.

All models are wrapped in a common interface.
External libraries (XGBoost, LightGBM) are optional dependencies.
"""

from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# Classification Registry
_CLASSIFICATION_REGISTRY: dict[str, tuple[type, dict]] = {
    "dummy_classifier": (
        DummyClassifier,
        {"strategy": "most_frequent"},
    ),
    "logistic_regression": (
        LogisticRegression,
        {"max_iter": 1000, "random_state": 42},
    ),
    "linear_svm": (
        LinearSVC,
        {"max_iter": 2000, "random_state": 42},
    ),
    "random_forest": (
        RandomForestClassifier,
        {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier,
        {"n_estimators": 100, "random_state": 42},
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier,
        {"random_state": 42},
    ),
    "knn": (
        KNeighborsClassifier,
        {"n_neighbors": 5, "n_jobs": -1},
    ),
}

if HAS_XGBOOST:
    _CLASSIFICATION_REGISTRY["xgboost"] = (
        XGBClassifier,
        {"n_estimators": 100, "random_state": 42, "eval_metric": "logloss"},
    )

if HAS_LIGHTGBM:
    _CLASSIFICATION_REGISTRY["lightgbm"] = (
        LGBMClassifier,
        {"n_estimators": 100, "random_state": 42, "verbose": -1},
    )


# Regression Registry
_REGRESSION_REGISTRY: dict[str, tuple[type, dict]] = {
    "dummy_regressor": (
        DummyRegressor,
        {"strategy": "mean"},
    ),
    "linear_regression": (
        LinearRegression,
        {},
    ),
    "ridge": (
        Ridge,
        {"random_state": 42},
    ),
    "random_forest_regressor": (
        RandomForestRegressor,
        {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    ),
    "gradient_boosting_regressor": (
        GradientBoostingRegressor,
        {"n_estimators": 100, "random_state": 42},
    ),
    "hist_gradient_boosting_regressor": (
        HistGradientBoostingRegressor,
        {"random_state": 42},
    ),
    "knn_regressor": (
        KNeighborsRegressor,
        {"n_neighbors": 5, "n_jobs": -1},
    ),
}

if HAS_XGBOOST:
    _REGRESSION_REGISTRY["xgboost_regressor"] = (
        XGBRegressor,
        {"n_estimators": 100, "random_state": 42},
    )

if HAS_LIGHTGBM:
    _REGRESSION_REGISTRY["lightgbm_regressor"] = (
        LGBMRegressor,
        {"n_estimators": 100, "random_state": 42, "verbose": -1},
    )


# Combined Registry
_REGISTRY: dict[str, tuple[type, dict]] = {
    **_CLASSIFICATION_REGISTRY,
    **_REGRESSION_REGISTRY,
}

# Baseline set guaranteed per project
BASELINE_MODELS = ["dummy_classifier", "logistic_regression", "random_forest"]
BASELINE_REGRESSION_MODELS = ["dummy_regressor", "linear_regression", "random_forest_regressor"]

# Models that do NOT support predict_proba natively
NO_PROBA_MODELS = {"linear_svm"}

# Mapping registry key → sklearn class name (for code generation)
SKLEARN_CLASS_MAP: dict[str, str] = {
    "dummy_classifier":                  "DummyClassifier",
    "logistic_regression":               "LogisticRegression",
    "linear_svm":                        "LinearSVC",
    "random_forest":                     "RandomForestClassifier",
    "gradient_boosting":                 "GradientBoostingClassifier",
    "hist_gradient_boosting":            "HistGradientBoostingClassifier",
    "knn":                               "KNeighborsClassifier",
    "dummy_regressor":                   "DummyRegressor",
    "linear_regression":                 "LinearRegression",
    "ridge":                             "Ridge",
    "random_forest_regressor":           "RandomForestRegressor",
    "gradient_boosting_regressor":       "GradientBoostingRegressor",
    "hist_gradient_boosting_regressor":  "HistGradientBoostingRegressor",
    "knn_regressor":                     "KNeighborsRegressor",
    "xgboost":                           "XGBClassifier",
    "xgboost_regressor":                 "XGBRegressor",
    "lightgbm":                          "LGBMClassifier",
    "lightgbm_regressor":                "LGBMRegressor",
}


def get_estimator(model_name: str, random_seed: int = 42, **overrides: Any) -> Any:
    """
    Return a fresh, unfitted estimator for the given model name.

    Args:
        model_name: Registry key (e.g. "logistic_regression" or "random_forest_regressor").
        random_seed: Overrides random_state where applicable.
        **overrides: Additional hyperparameter overrides.

    Returns:
        Sklearn-compatible estimator.

    Raises:
        ValueError: If model_name is not in the registry.
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. Available: {list(_REGISTRY.keys())}"
        )

    cls, default_params = _REGISTRY[model_name]
    params = dict(default_params)

    if "random_state" in params:
        params["random_state"] = random_seed

    params.update(overrides)
    params.pop("task_type", None)
    return cls(**params)


def list_models(task_type: str = "classification") -> list[str]:
    """Return available model keys based on task_type."""
    if task_type in ("regression", "regressor"):
        return list(_REGRESSION_REGISTRY.keys())
    return list(_CLASSIFICATION_REGISTRY.keys())


def is_regression_model(model_name: str) -> bool:
    """Return True if model_name is in regression registry."""
    return model_name in _REGRESSION_REGISTRY


def supports_proba(model_name: str) -> bool:
    return model_name not in NO_PROBA_MODELS and not is_regression_model(model_name)
