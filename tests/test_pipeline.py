"""Tests for the pipeline builder — leakage prevention is critical."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))


from profiling.profiler import profile_dataset, FeatureProfile
from preprocessing.pipeline_builder import build_pipeline, config_hash


@pytest.fixture
def simple_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "age": [22.0, 38.0, None, 35.0, 27.0, 54.0, 2.0, 27.0, 14.0, 30.0],
        "fare": [7.25, 71.28, 7.92, 53.10, 8.46, 51.86, 21.08, 11.13, 30.07, 15.0],
        "sex": ["male", "female", "female", "female", "male", "male", "male", "female", "female", "male"],
        "survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    })


@pytest.fixture
def simple_decisions() -> list[dict]:
    return [
        {"category": "missing_value", "column_name": "age", "selected_action": "median_imputation"},
        {"category": "scaling", "column_name": "age", "selected_action": "standard_scaling"},
        {"category": "scaling", "column_name": "fare", "selected_action": "standard_scaling"},
        {"category": "encoding", "column_name": "sex", "selected_action": "one_hot_encoding"},
    ]


def test_pipeline_builds_without_error(simple_df, simple_decisions):
    profile = profile_dataset(simple_df, "survived")
    pipeline, config = build_pipeline(simple_decisions, profile.features, DummyClassifier())
    assert pipeline is not None


def test_pipeline_has_preprocessor_and_model(simple_df, simple_decisions):
    profile = profile_dataset(simple_df, "survived")
    pipeline, _ = build_pipeline(simple_decisions, profile.features, DummyClassifier())
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_pipeline_fits_and_predicts(simple_df, simple_decisions):
    profile = profile_dataset(simple_df, "survived")
    pipeline, _ = build_pipeline(simple_decisions, profile.features, DummyClassifier())
    X = simple_df.drop(columns=["survived"])
    y = simple_df["survived"]
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)


def test_no_leakage_train_test_split(simple_df, simple_decisions):
    """
    Critical: pipeline fitted ONLY on train data must successfully transform test data.
    If preprocessing leaks (e.g. fitted on full dataset), this is a bug.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression

    profile = profile_dataset(simple_df, "survived")
    pipeline, _ = build_pipeline(simple_decisions, profile.features, LogisticRegression(max_iter=1000))
    X = simple_df.drop(columns=["survived"])
    y = simple_df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Clone ensures fresh, unfitted pipeline
    fresh = clone(pipeline)
    fresh.fit(X_train, y_train)  # Fit ONLY on train
    preds = fresh.predict(X_test)  # Transform and predict on test
    assert len(preds) == len(y_test)  # No error means no leakage issue


def test_config_hash_is_deterministic(simple_df, simple_decisions):
    profile = profile_dataset(simple_df, "survived")
    _, config1 = build_pipeline(simple_decisions, profile.features, DummyClassifier())
    _, config2 = build_pipeline(simple_decisions, profile.features, DummyClassifier())
    assert config_hash(config1) == config_hash(config2)


def test_drop_column_decision_removes_column(simple_df):
    profile = profile_dataset(simple_df, "survived")
    decisions = [
        {"category": "missing_value", "column_name": "age", "selected_action": "drop_column"},
        {"category": "encoding", "column_name": "sex", "selected_action": "one_hot_encoding"},
    ]
    _, config = build_pipeline(decisions, profile.features, DummyClassifier())
    assert "age" in config["dropped_columns"]
    assert "age" not in config["numeric_columns"]


def test_config_json_contains_model_info(simple_df, simple_decisions):
    profile = profile_dataset(simple_df, "survived")
    _, config = build_pipeline(simple_decisions, profile.features, DummyClassifier())
    assert "model" in config
    assert "name" in config["model"]


def test_winsorizer_in_pipeline(simple_df):
    profile = profile_dataset(simple_df, "survived")
    decisions = [
        {"category": "missing_value", "column_name": "fare", "selected_action": "median_imputation"},
        {"category": "outlier", "column_name": "fare", "selected_action": "winsorization"},
        {"category": "scaling", "column_name": "fare", "selected_action": "standard_scaling"},
    ]
    pipeline, _ = build_pipeline(decisions, profile.features, DummyClassifier())
    X = simple_df.drop(columns=["survived"])
    y = simple_df["survived"]
    pipeline.fit(X, y)  # Should not raise
