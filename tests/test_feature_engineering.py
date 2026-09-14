"""
Unit tests for AutoFeatureEngineer transformer and pipeline integration.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from ml_engine.preprocessing.feature_engineering import AutoFeatureEngineer
from ml_engine.preprocessing.pipeline_builder import build_pipeline


def test_auto_feature_engineer_transform():
    # Synthetic dataset with 1 skewed column (exponential) and 1 non-skewed column (normal)
    np.random.seed(42)
    skewed_col = np.random.exponential(scale=2.0, size=100)
    normal_col = np.random.normal(loc=10.0, scale=1.0, size=100)
    
    X = np.column_stack([skewed_col, normal_col])
    
    fe = AutoFeatureEngineer(skew_threshold=1.0, create_interactions=True, max_interactions=2)
    X_trans = fe.fit_transform(X)
    
    # 2 original features + 1 interaction feature = 3 features total
    assert X_trans.shape[1] == 3
    # Skewed column index should be 0
    assert 0 in fe.skewed_cols_idx_


def test_build_pipeline_with_feature_engineering():
    from ml_engine.profiling.profiler import FeatureProfile
    
    profiles = [
        FeatureProfile(name="num1", dtype="numeric", missing_count=0, missing_rate=0.0, unique_count=50, constant=False),
        FeatureProfile(name="num2", dtype="numeric", missing_count=0, missing_rate=0.0, unique_count=50, constant=False),
    ]
    decisions = [
        {"column_name": "num1", "category": "scaling", "selected_action": "standard_scaling"},
        {"column_name": "num2", "category": "scaling", "selected_action": "standard_scaling"},
    ]
    
    pipeline, config = build_pipeline(
        decisions=decisions,
        profile_features=profiles,
        model_estimator=LinearRegression(),
        enable_feature_engineering=True,
    )
    
    assert "feature_engineer" in pipeline.named_steps
    assert config["enable_feature_engineering"] is True


def test_interaction_avoids_highly_collinear_pairs():
    # Fitur 0 dan 1 berkorelasi sangat tinggi (~0.99)
    # Fitur 0 dan 2 berkorelasi moderat (~0.4)
    np.random.seed(42)
    x0 = np.random.normal(0, 1, 200)
    x1 = x0 + np.random.normal(0, 0.05, 200)  # corr ~ 0.99
    x2 = 0.4 * x0 + np.random.normal(0, 0.9, 200)  # corr ~ 0.4
    X = np.column_stack([x0, x1, x2])

    fe = AutoFeatureEngineer(create_interactions=True, max_interactions=1)
    fe.fit(X)

    # Pasangan (0, 1) yang multikolinear tinggi TIDAK boleh dipilih
    assert (0, 1) not in fe.interaction_pairs_idx_
    assert (1, 0) not in fe.interaction_pairs_idx_
    assert len(fe.interaction_pairs_idx_) == 1
