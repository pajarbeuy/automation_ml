import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml_engine.experiments.tuner import run_hyperparameter_search


def test_hyperparameter_search_random_forest():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        "target": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    })
    X = df.drop(columns=["target"])
    y = df["target"]

    pipeline = Pipeline([
        ("model", RandomForestClassifier(random_state=42))
    ])

    res = run_hyperparameter_search(
        pipeline=pipeline,
        X=X,
        y=y,
        model_name="random_forest",
        n_trials=3,
        random_seed=42,
        primary_metric="accuracy",
        cv_folds=2,
    )

    assert res.model_name == "random_forest"
    assert res.best_score > 0
    assert len(res.trials) == 3
    assert "n_estimators" in res.best_params or "max_depth" in res.best_params
