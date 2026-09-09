from __future__ import annotations

import sys
from pathlib import Path

# Add packages/ml_engine to sys.path if not present
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))

from export.code_generator import generate_pipeline_script
from export.report_generator import generate_html_report
from tracking.mlflow_tracker import is_mlflow_available, log_experiment_to_mlflow


def test_code_generator_returns_valid_python():
    config = {
        "numeric_config": {
            "age": {"missing": "median_imputation", "outlier": "winsorization", "scaling": "standard_scaling"}
        },
        "categorical_config": {
            "gender": {"missing": "most_frequent_imputation", "encoding": "one_hot_encoding"}
        },
        "dropped_columns": ["id"],
    }
    script = generate_pipeline_script(
        config=config,
        model_name="RandomForestClassifier",
        seed=42,
        experiment_id="exp-12345678",
        dataset_filename="test_dataset.csv",
        target_column="label",
    )

    assert "import pandas as pd" in script
    assert "RandomForestClassifier" in script
    assert "ColumnTransformer" in script
    assert "num_age" in script
    assert "cat_gender" in script
    assert "Winsorizer" in script
    assert 'TARGET = "label"' in script


def test_code_generator_regression_and_tuning():
    config = {
        "numeric_config": {
            "house_size": {"missing": "median_imputation", "scaling": "standard_scaling"}
        },
        "enable_feature_engineering": True,
        "is_tuned": True,
        "model_params": {"n_estimators": 50, "max_depth": 5},
    }
    script = generate_pipeline_script(
        config=config,
        model_name="RandomForestRegressor",
        seed=42,
        experiment_id="exp-reg-1",
        dataset_filename="houses.csv",
        target_column="price",
        task_type="regression",
    )

    assert "RandomForestRegressor" in script
    assert "from sklearn.ensemble import RandomForestRegressor" in script
    assert "max_depth=5" in script
    assert "n_estimators=50" in script
    assert "AutoFeatureEngineer" in script
    assert "stratify=y" not in script
    assert "KFold" in script
    assert "r2_score" in script
    assert "TARGET = \"price\"" in script


def test_report_generator_returns_valid_html():
    profile = {
        "duplicate_rows": 0,
        "features": [
            {"name": "age", "dtype": "numeric", "missing_count": 0, "missing_rate": 0.0, "unique_count": 50, "constant": False},
            {"name": "gender", "dtype": "categorical", "missing_count": 2, "missing_rate": 0.02, "unique_count": 2, "constant": False},
        ],
    }
    decisions = [
        {"column_name": "gender", "category": "missing_values", "selected_action": "most_frequent_imputation", "status": "accepted"}
    ]
    experiments = [
        {
            "model_name": "RandomForestClassifier",
            "status": "completed",
            "training_time_seconds": 1.2,
            "random_seed": 42,
            "metrics": [{"name": "f1_mean", "value": 0.92, "split": "cv"}, {"name": "accuracy_mean", "value": 0.94, "split": "cv"}],
        }
    ]

    html = generate_html_report(
        project_name="Test Project",
        project_id="proj-123",
        target_column="target",
        dataset_filename="data.csv",
        dataset_rows=100,
        dataset_columns=3,
        profile=profile,
        decisions=decisions,
        experiments=experiments,
        candidate_experiment=experiments[0],
        confusion_matrix=[[10, 2], [1, 15]],
        generated_at="2026-09-05 22:00 UTC",
    )

    assert "<!DOCTYPE html>" in html
    assert "Test Project" in html
    assert "RandomForestClassifier" in html
    assert "Confusion Matrix" in html
    assert "92.00%" in html


def test_report_generator_regression():
    profile = {
        "duplicate_rows": 0,
        "features": [
            {"name": "size", "dtype": "numeric", "missing_count": 0, "missing_rate": 0.0, "unique_count": 50, "constant": False},
        ],
    }
    experiments = [
        {
            "model_name": "RandomForestRegressor",
            "status": "completed",
            "training_time_seconds": 0.8,
            "random_seed": 42,
            "metrics": [
                {"name": "r2_mean", "value": 0.885, "split": "cv"},
                {"name": "rmse_mean", "value": 3.25, "split": "cv"},
                {"name": "mae_mean", "value": 2.10, "split": "cv"},
            ],
        }
    ]

    html = generate_html_report(
        project_name="House Prices",
        project_id="proj-reg",
        target_column="price",
        dataset_filename="houses.csv",
        dataset_rows=200,
        dataset_columns=2,
        profile=profile,
        decisions=[],
        experiments=experiments,
        candidate_experiment=experiments[0],
        generated_at="2026-09-09 20:00 UTC",
        task_type="regression",
        residuals_summary={"Mean Error": -0.02, "Std Error": 1.15},
    )

    assert "<!DOCTYPE html>" in html
    assert "House Prices" in html
    assert "RandomForestRegressor" in html
    assert "R² (CV)" in html
    assert "0.8850" in html
    assert "3.2500" in html
    assert "2.1000" in html
    assert "Mean Error" in html
    assert "Confusion Matrix" not in html


def test_mlflow_tracker_graceful_handling():
    # Should not raise exception even if mlflow is not installed
    avail = is_mlflow_available()
    assert isinstance(avail, bool)

    # Calling log_experiment_to_mlflow with dummy record should handle missing mlflow or server gracefully
    class DummyRecord:
        experiment_id = "exp-1"
        project_id = "proj-1"
        dataset_id = "ds-1"
        model_name = "RandomForestClassifier"
        random_seed = 42
        cv_folds = 5
        primary_metric = "f1"
        pipeline_hash = "abc12345"
        status = "completed"
        config_json = {}
        metrics_flat = {"f1_mean": 0.9}
        training_time_seconds = 1.0
        artifacts = []

    result = log_experiment_to_mlflow(DummyRecord())
    # If mlflow is not installed, result is False (graceful fallback)
    assert isinstance(result, bool)
