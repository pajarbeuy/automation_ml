"""Tests for the profiling engine."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))

from profiling.profiler import profile_dataset, _detect_dtype


@pytest.fixture
def titanic_like_df() -> pd.DataFrame:
    """Minimal Titanic-like dataset for testing."""
    return pd.DataFrame(
        {
            "age": [22.0, 38.0, None, 35.0, None, 27.0, 54.0, 2.0, 27.0, 14.0],
            "fare": [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
            "sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"],
            "pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
            "survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        }
    )


def test_profile_returns_correct_row_count(titanic_like_df):
    profile = profile_dataset(titanic_like_df, "survived")
    assert profile.rows == 10


def test_profile_correct_column_count(titanic_like_df):
    profile = profile_dataset(titanic_like_df, "survived")
    assert profile.columns == 5  # all including target


def test_profile_missing_value_detection(titanic_like_df):
    profile = profile_dataset(titanic_like_df, "survived")
    age_feature = next(f for f in profile.features if f.name == "age")
    assert age_feature.missing_count == 2
    assert abs(age_feature.missing_rate - 0.2) < 0.01


def test_profile_dtype_detection(titanic_like_df):
    profile = profile_dataset(titanic_like_df, "survived")
    sex_feature = next(f for f in profile.features if f.name == "sex")
    age_feature = next(f for f in profile.features if f.name == "age")
    assert sex_feature.dtype == "categorical"
    assert age_feature.dtype == "numeric"


def test_profile_numeric_stats(titanic_like_df):
    profile = profile_dataset(titanic_like_df, "survived")
    fare_feature = next(f for f in profile.features if f.name == "fare")
    assert fare_feature.mean is not None
    assert fare_feature.std is not None
    assert fare_feature.p50 is not None


def test_profile_class_distribution(titanic_like_df):
    profile = profile_dataset(titanic_like_df, "survived")
    assert "0" in profile.class_distribution or 0 in profile.class_distribution
    total = sum(profile.class_distribution.values())
    assert abs(total - 1.0) < 0.01  # proportions sum to 1


def test_profile_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "target": [0, 0, 1, 1]})
    profile = profile_dataset(df, "target")
    assert profile.duplicate_rows == 1


def test_no_modification_of_input(titanic_like_df):
    original_na_count = titanic_like_df["age"].isna().sum()
    profile = profile_dataset(titanic_like_df, "survived")
    assert titanic_like_df["age"].isna().sum() == original_na_count  # unchanged


def test_detect_dtype_numeric():
    s = pd.Series([1.0, 2.5, 3.1])
    s.name = "col"
    assert _detect_dtype(s) == "numeric"


def test_detect_dtype_categorical():
    s = pd.Series(["a", "b", "c", "a"])
    s.name = "col"
    assert _detect_dtype(s) == "categorical"


def test_detect_dtype_boolean():
    s = pd.Series([True, False, True, False])
    s.name = "col"
    assert _detect_dtype(s) == "boolean"


def test_constant_column_detection():
    df = pd.DataFrame({"a": [1, 1, 1, 1], "label": [0, 1, 0, 1]})
    profile = profile_dataset(df, "label")
    a_feature = next(f for f in profile.features if f.name == "a")
    assert a_feature.constant is True


def test_dataset_hash_deterministic(titanic_like_df):
    h1 = profile_dataset(titanic_like_df, "survived").dataset_hash
    h2 = profile_dataset(titanic_like_df, "survived").dataset_hash
    assert h1 == h2


def test_detect_dtype_identifier_string():
    # 100 baris dengan 96 nilai unik (96% > 95% threshold) dan keyword 'id' di nama
    vals = [f"usr_{i}" for i in range(96)] + ["usr_0", "usr_1", "usr_2", "usr_3"]
    s = pd.Series(vals, name="user_id")
    assert _detect_dtype(s) == "identifier"

    # Data yang sama tapi nama bukan keyword ID -> categorical
    s2 = pd.Series(vals, name="description")
    assert _detect_dtype(s2) == "categorical"
