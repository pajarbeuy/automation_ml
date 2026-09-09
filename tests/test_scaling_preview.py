import pandas as pd
import pytest
from ml_engine.eda.charts import scaling_preview
from ml_engine.profiling.profiler import profile_dataset


def test_scaling_preview_returns_valid_structure():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        "income": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 100000],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })
    profile = profile_dataset(df, target_column="category")
    decisions = [
        {"column_name": "age", "category": "scaling", "selected_action": "standard_scaling"},
        {"column_name": "income", "category": "scaling", "selected_action": "robust_scaling"},
    ]

    res = scaling_preview(df, decisions, profile)

    assert "age" in res
    assert "income" in res
    assert res["age"]["scaling_action"] == "standard_scaling"
    assert "raw" in res["age"]
    assert "transformed" in res["age"]
    assert len(res["age"]["raw"]["bins"]) > 0
    assert len(res["age"]["transformed"]["bins"]) > 0
    # StandardScaler transforms mean to ~0
    assert abs(res["age"]["transformed"]["mean"]) < 1e-4


def test_profile_dataset_default_target_column():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    prof = profile_dataset(df)
    assert prof.rows == 3
    assert prof.columns == 2
