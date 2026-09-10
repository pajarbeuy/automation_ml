"""Tests for the recommendation engine."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))


from profiling.profiler import profile_dataset
from recommendations.engine import generate_recommendations


@pytest.fixture
def skewed_numeric_df() -> pd.DataFrame:
    """Dataset with skewed numeric, missing values, and imbalanced target."""
    import numpy as np
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "income": [None if i % 10 == 0 else float(v) for i, v in enumerate(np.random.exponential(50000, n))],
            "age": np.random.normal(35, 10, n),
            "city": np.random.choice(["NY", "LA", "Chicago", "Houston", "Phoenix"], n),
            "label": [1 if i < 20 else 0 for i in range(n)],  # 10% positive
        }
    )


def test_recommendations_generated(skewed_numeric_df):
    profile = profile_dataset(skewed_numeric_df, "label")
    recs = generate_recommendations(skewed_numeric_df, profile)
    assert len(recs) > 0


def test_missing_value_recommendation_for_skewed(skewed_numeric_df):
    profile = profile_dataset(skewed_numeric_df, "label")
    recs = generate_recommendations(skewed_numeric_df, profile)
    missing_recs = [r for r in recs if r.category == "missing_value" and r.column == "income"]
    assert len(missing_recs) == 1
    # Skewed distribution → median should be recommended
    assert missing_recs[0].recommended_action == "median_imputation"


def test_class_imbalance_detected(skewed_numeric_df):
    profile = profile_dataset(skewed_numeric_df, "label")
    recs = generate_recommendations(skewed_numeric_df, profile)
    imbalance_recs = [r for r in recs if r.category == "class_imbalance"]
    assert len(imbalance_recs) == 1
    assert imbalance_recs[0].recommended_action == "use_class_weight"


def test_encoding_recommendation_for_categorical(skewed_numeric_df):
    profile = profile_dataset(skewed_numeric_df, "label")
    recs = generate_recommendations(skewed_numeric_df, profile)
    enc_recs = [r for r in recs if r.category == "encoding" and r.column == "city"]
    assert len(enc_recs) == 1
    assert enc_recs[0].recommended_action == "one_hot_encoding"


def test_scaling_recommendation_for_numeric(skewed_numeric_df):
    profile = profile_dataset(skewed_numeric_df, "label")
    recs = generate_recommendations(skewed_numeric_df, profile)
    scale_recs = [r for r in recs if r.category == "scaling"]
    assert len(scale_recs) >= 1


def test_high_cardinality_warning():
    import numpy as np
    df = pd.DataFrame({
        "city": [f"city_{i}" for i in range(200)],  # 200 unique values
        "price": np.random.uniform(10, 1000, 200),
        "label": np.random.randint(0, 2, 200),
    })
    profile = profile_dataset(df, "label")
    recs = generate_recommendations(df, profile)
    city_enc = next((r for r in recs if r.category == "encoding" and r.column == "city"), None)
    assert city_enc is not None
    # High cardinality → should warn / not recommend one-hot
    assert city_enc.recommended_action != "one_hot_encoding"
    assert any("high_cardinality" in e for e in city_enc.evidence)


def test_all_recommendations_have_required_fields(skewed_numeric_df):
    profile = profile_dataset(skewed_numeric_df, "label")
    recs = generate_recommendations(skewed_numeric_df, profile)
    for rec in recs:
        assert rec.id
        assert rec.category
        assert rec.recommended_action
        assert 0.0 <= rec.confidence <= 1.0
        assert 0.0 <= rec.score <= 100.0
        assert len(rec.evidence) > 0
        assert rec.rationale
        assert isinstance(rec.alternatives, list)
        assert rec.status == "proposed"


def test_recommendations_sorted_by_score_descending(skewed_numeric_df):
    profile = profile_dataset(skewed_numeric_df, "label")
    recs = generate_recommendations(skewed_numeric_df, profile)
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_no_recommendations_for_clean_dataset():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": ["x", "y", "x", "y", "x"],
        "label": [0, 1, 0, 1, 0],
    })
    profile = profile_dataset(df, "label")
    recs = generate_recommendations(df, profile)
    # No missing values → no missing_value recommendations
    missing_recs = [r for r in recs if r.category == "missing_value"]
    assert len(missing_recs) == 0


def test_regression_no_class_imbalance(skewed_numeric_df):
    """Ensure that regression tasks NEVER produce class_imbalance recommendations."""
    profile = profile_dataset(skewed_numeric_df, "label")
    recs = generate_recommendations(skewed_numeric_df, profile, task_type="regression")
    imbalance_recs = [r for r in recs if r.category == "class_imbalance"]
    assert len(imbalance_recs) == 0


def test_regression_target_skewness_and_outliers():
    """Verify that a skewed continuous target produces target_transformation and target_outliers."""
    import numpy as np
    np.random.seed(42)
    n = 200
    # Right-skewed housing prices with some extreme outliers
    prices = np.random.exponential(scale=500000, size=n)
    prices[0] = 15000000.0  # extreme outlier
    prices[1] = 20000000.0  # extreme outlier
    prices[2] = 25000000.0  # extreme outlier
    df = pd.DataFrame({
        "sqft": np.random.normal(1500, 300, n),
        "bedrooms": np.random.choice([1, 2, 3, 4], n),
        "price": prices,
    })
    profile = profile_dataset(df, "price")
    recs = generate_recommendations(df, profile, task_type="regression")

    # No class imbalance
    assert not any(r.category == "class_imbalance" for r in recs)

    # Target skewness recommendation should exist
    skew_recs = [r for r in recs if r.category == "target_transformation"]
    assert len(skew_recs) == 1
    assert skew_recs[0].recommended_action == "log1p_transform"
    assert skew_recs[0].column == "price"

    # Target outliers recommendation should exist
    outlier_recs = [r for r in recs if r.category == "target_outliers"]
    assert len(outlier_recs) == 1
    assert outlier_recs[0].recommended_action == "use_robust_models"

