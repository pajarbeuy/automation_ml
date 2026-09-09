"""Tests for the experiment runner."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))


from profiling.profiler import profile_dataset
from experiments.runner import run_experiment


@pytest.fixture
def sample_csv(tmp_path) -> tuple[str, str]:
    """Create a temp CSV with classification data."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "age": np.random.normal(35, 10, n),
        "income": np.random.exponential(50000, n),
        "sex": np.random.choice(["male", "female"], n),
        "label": np.random.randint(0, 2, n),
    })
    path = str(tmp_path / "test_data.csv")
    df.to_csv(path, index=False)
    return path, "label"


@pytest.fixture
def storage_dir(tmp_path) -> str:
    storage = tmp_path / "storage"
    storage.mkdir()
    return str(storage)


@pytest.fixture
def decisions_and_features(sample_csv):
    csv_path, target = sample_csv
    df = pd.read_csv(csv_path)
    profile = profile_dataset(df, target)
    decisions = [
        {"category": "missing_value", "column_name": "age", "selected_action": "median_imputation"},
        {"category": "scaling", "column_name": "age", "selected_action": "standard_scaling"},
        {"category": "scaling", "column_name": "income", "selected_action": "robust_scaling"},
        {"category": "encoding", "column_name": "sex", "selected_action": "one_hot_encoding"},
    ]
    return decisions, profile.features


def test_experiment_completes(sample_csv, storage_dir, decisions_and_features):
    csv_path, target = sample_csv
    decisions, features = decisions_and_features
    record = run_experiment(
        experiment_id="test-exp-001",
        project_id="test-project",
        dataset_id="test-dataset",
        dataset_path=csv_path,
        target_column=target,
        model_name="logistic_regression",
        decisions=decisions,
        profile_features=features,
        storage_path=storage_dir,
        random_seed=42,
        cv_folds=3,
        primary_metric="f1",
    )
    assert record.status == "completed"


def test_experiment_produces_metrics(sample_csv, storage_dir, decisions_and_features):
    csv_path, target = sample_csv
    decisions, features = decisions_and_features
    record = run_experiment(
        experiment_id="test-exp-002",
        project_id="test-project",
        dataset_id="test-dataset",
        dataset_path=csv_path,
        target_column=target,
        model_name="random_forest",
        decisions=decisions,
        profile_features=features,
        storage_path=storage_dir,
        cv_folds=3,
    )
    assert "f1_mean" in record.metrics_flat
    assert "accuracy_mean" in record.metrics_flat
    assert 0.0 <= record.metrics_flat["accuracy_mean"] <= 1.0


def test_experiment_saves_artifacts(sample_csv, storage_dir, decisions_and_features):
    csv_path, target = sample_csv
    decisions, features = decisions_and_features
    record = run_experiment(
        experiment_id="test-exp-003",
        project_id="test-project",
        dataset_id="test-dataset",
        dataset_path=csv_path,
        target_column=target,
        model_name="logistic_regression",
        decisions=decisions,
        profile_features=features,
        storage_path=storage_dir,
        cv_folds=3,
    )
    assert len(record.artifacts) >= 2
    for art in record.artifacts:
        assert Path(art["path"]).exists()


def test_experiment_fails_gracefully_on_bad_path(storage_dir, decisions_and_features):
    _, features = decisions_and_features
    record = run_experiment(
        experiment_id="test-exp-004",
        project_id="test-project",
        dataset_id="test-dataset",
        dataset_path="/nonexistent/path/data.csv",
        target_column="label",
        model_name="logistic_regression",
        decisions=[],
        profile_features=features,
        storage_path=storage_dir,
    )
    assert record.status == "failed"
    assert "LOAD_ERROR" in record.error_message


def test_experiment_records_environment(sample_csv, storage_dir, decisions_and_features):
    csv_path, target = sample_csv
    decisions, features = decisions_and_features
    record = run_experiment(
        experiment_id="test-exp-005",
        project_id="test-project",
        dataset_id="test-dataset",
        dataset_path=csv_path,
        target_column=target,
        model_name="dummy_classifier",
        decisions=decisions,
        profile_features=features,
        storage_path=storage_dir,
        cv_folds=3,
    )
    assert "python_version" in record.environment
    assert "sklearn_version" in record.environment


def test_experiment_reproducibility(sample_csv, storage_dir, decisions_and_features):
    """Same seed → same CV metrics."""
    csv_path, target = sample_csv
    decisions, features = decisions_and_features
    kwargs = dict(
        project_id="p", dataset_id="d", dataset_path=csv_path,
        target_column=target, model_name="logistic_regression",
        decisions=decisions, profile_features=features,
        storage_path=storage_dir, random_seed=42, cv_folds=3,
    )
    r1 = run_experiment(experiment_id="rep-001", **kwargs)
    r2 = run_experiment(experiment_id="rep-002", **kwargs)
    assert r1.metrics_flat.get("f1_mean") == r2.metrics_flat.get("f1_mean")
