from __future__ import annotations

import pytest
import numpy as np
from recommendations.meta_ranker import (
    MetaFeatureExtractor,
    StrategyEncoder,
    MetaLearningRanker,
)


def test_meta_feature_extraction():
    profile = {
        "rows": 1000,
        "columns": 8,
        "features": [
            {"dtype": "numeric", "missing_rate": 0.05, "skewness": 2.1, "has_outliers": True},
            {"dtype": "numeric", "missing_rate": 0.0, "skewness": -0.3, "has_outliers": False},
            {"dtype": "numeric", "missing_rate": 0.0, "skewness": 0.0, "has_outliers": False},
            {"dtype": "categorical", "missing_rate": 0.02, "skewness": None, "has_outliers": False},
            {"dtype": "categorical", "missing_rate": 0.0, "skewness": None, "has_outliers": False},
        ],
        "target_summary": {
            "class_distribution": {"0": 800, "1": 200}
        },
    }

    mf = MetaFeatureExtractor.extract(profile, task_type="classification")
    assert mf["log_rows"] == 3.0  # log10(1000)
    assert mf["is_classification"] == 1.0
    assert mf["is_regression"] == 0.0
    assert mf["imbalance_ratio"] == 0.25  # 200 / 800
    assert mf["outlier_ratio"] > 0.0

    vec = MetaFeatureExtractor.to_vector(mf)
    assert len(vec) == len(MetaFeatureExtractor.FEATURE_NAMES)
    assert isinstance(vec, np.ndarray)


def test_strategy_encoder():
    rec = StrategyEncoder.encode("recommended")
    base = StrategyEncoder.encode("baseline")
    alt = StrategyEncoder.encode("alternative")

    assert np.array_equal(rec, [1.0, 0.0, 0.0])
    assert np.array_equal(base, [0.0, 1.0, 0.0])
    assert np.array_equal(alt, [0.0, 0.0, 1.0])


def test_meta_learning_ranker_cold_start():
    ranker = MetaLearningRanker()
    assert ranker.is_trained is False

    meta_features = {"log_rows": 2.5, "numeric_ratio": 0.8, "is_classification": 1.0}
    candidates = [
        {"candidate_id": "c1", "name": "Recommended Strategy", "strategy_type": "recommended", "heuristic_score": 90.0},
        {"candidate_id": "c2", "name": "Baseline Strategy", "strategy_type": "baseline", "heuristic_score": 70.0},
    ]

    ranked = ranker.rank_candidates(meta_features, candidates)
    assert len(ranked) == 2
    assert ranked[0]["candidate_id"] == "c1"  # Recommended ranked above baseline
    assert "blended_rank_score" in ranked[0]
    assert "Cold-start prior" in ranked[0]["rationale"]


def test_meta_learning_ranker_training_and_lift_prediction():
    ranker = MetaLearningRanker()

    # Generate synthetic historical benchmark records
    records = []
    rng = np.random.RandomState(42)
    for _ in range(15):
        mf = {
            "log_rows": float(rng.uniform(2.0, 5.0)),
            "log_cols": float(rng.uniform(0.5, 2.0)),
            "ratio_rows_cols": float(rng.uniform(10.0, 500.0)),
            "numeric_ratio": float(rng.uniform(0.3, 0.9)),
            "categorical_ratio": float(rng.uniform(0.1, 0.7)),
            "missing_cells_ratio": float(rng.uniform(0.0, 0.2)),
            "max_missing_ratio": float(rng.uniform(0.0, 0.5)),
            "mean_abs_skewness": float(rng.uniform(0.0, 3.0)),
            "max_abs_skewness": float(rng.uniform(0.5, 5.0)),
            "outlier_ratio": float(rng.uniform(0.0, 0.4)),
            "is_regression": 0.0,
            "is_classification": 1.0,
            "imbalance_ratio": float(rng.uniform(0.1, 1.0)),
        }
        # Simulated truth: recommended strategy gains ~ +3% lift
        records.append({"meta_features": mf, "strategy_name": "recommended", "lift": 0.035 + rng.normal(0, 0.005)})
        records.append({"meta_features": mf, "strategy_name": "baseline", "lift": 0.0})
        records.append({"meta_features": mf, "strategy_name": "alternative", "lift": 0.015 + rng.normal(0, 0.005)})

    res = ranker.train(records)
    assert res["status"] == "trained"
    assert ranker.is_trained is True

    test_mf = records[0]["meta_features"]
    pred_lift_rec, conf_rec = ranker.predict_lift(test_mf, "recommended")
    pred_lift_base, conf_base = ranker.predict_lift(test_mf, "baseline")

    assert pred_lift_rec > pred_lift_base
    assert conf_rec >= 50.0

    candidates = [
        {"candidate_id": "b1", "name": "Baseline", "strategy_type": "baseline", "heuristic_score": 70.0},
        {"candidate_id": "r1", "name": "Recommended", "strategy_type": "recommended", "heuristic_score": 85.0},
    ]
    ranked = ranker.rank_candidates(test_mf, candidates)
    assert ranked[0]["candidate_id"] == "r1"
    assert "Surrogate meta-model" in ranked[0]["rationale"]
