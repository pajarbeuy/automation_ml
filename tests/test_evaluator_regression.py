"""
Unit tests for regression evaluator, runner, and tuner.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from ml_engine.evaluation.evaluator import run_cross_validation, EvaluationResult
from ml_engine.experiments.tuner import run_hyperparameter_search


def test_regression_cross_validation():
    np.random.seed(42)
    X = pd.DataFrame({
        "x1": np.random.randn(100),
        "x2": np.random.randn(100),
    })
    y = pd.Series(2.0 * X["x1"] - 3.0 * X["x2"] + np.random.randn(100) * 0.1, name="target")

    pipe = Pipeline([("model", LinearRegression())])
    res: EvaluationResult = run_cross_validation(
        pipeline=pipe,
        X=X,
        y=y,
        model_name="linear_regression",
        n_splits=3,
        task_type="regression",
    )

    assert res.model_name == "linear_regression"
    assert "r2_mean" in res.summary
    assert "mae_mean" in res.summary
    assert "rmse_mean" in res.summary
    assert res.summary["r2_mean"] > 0.90
    assert res.residual_data is not None
    assert len(res.residual_data) > 0


def test_regression_hyperparameter_search():
    np.random.seed(42)
    X = pd.DataFrame({
        "x1": np.random.randn(50),
        "x2": np.random.randn(50),
    })
    y = pd.Series(1.5 * X["x1"] + np.random.randn(50) * 0.2, name="target")

    pipe = Pipeline([("model", Ridge())])
    res = run_hyperparameter_search(
        pipeline=pipe,
        X=X,
        y=y,
        model_name="ridge",
        n_trials=5,
        primary_metric="r2",
        cv_folds=3,
        task_type="regression",
    )

    assert res.model_name == "ridge"
    assert "alpha" in res.best_params
    assert len(res.trials) > 0
