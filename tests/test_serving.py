from __future__ import annotations

import sys
from pathlib import Path

# Add packages/ml_engine to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from serving.inference import predict


def test_serving_inference_classification_single_and_batch():
    # 1. Train a mock classification pipeline on raw features
    df = pd.DataFrame({
        "age": [25, 45, 35, 50, 23],
        "gender": ["male", "female", "female", "male", "female"],
        "target": [0, 1, 0, 1, 0],
    })

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["age"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["gender"]),
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(random_state=42)),
    ])
    pipeline.fit(df[["age", "gender"]], df["target"])

    # 2. Test single raw dict input
    raw_input = {"age": 28, "gender": "male"}
    res_single = predict(pipeline, raw_input, task_type="classification")

    assert len(res_single) == 1
    assert "prediction" in res_single[0]
    assert res_single[0]["prediction"] in [0, 1]
    assert "confidence" in res_single[0]
    assert 0.0 <= res_single[0]["confidence"] <= 1.0
    assert "probabilities" in res_single[0]

    # 3. Test batch list of dicts
    batch_input = [
        {"age": 22, "gender": "female"},
        {"age": 60, "gender": "male"},
    ]
    res_batch = predict(pipeline, batch_input, task_type="classification")
    assert len(res_batch) == 2
    for r in res_batch:
        assert "prediction" in r
        assert "confidence" in r


def test_serving_inference_regression_single_and_batch():
    # 1. Train a mock regression pipeline on raw features
    df = pd.DataFrame({
        "sqft": [800, 1200, 1500, 2000, 2500],
        "rooms": [2, 3, 3, 4, 5],
        "price": [150000.0, 220000.0, 280000.0, 360000.0, 450000.0],
    })

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["sqft", "rooms"]),
        ]
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])
    pipeline.fit(df[["sqft", "rooms"]], df["price"])

    # 2. Test single raw dict input
    raw_input = {"sqft": 1400, "rooms": 3}
    res_single = predict(pipeline, raw_input, task_type="regression")

    assert len(res_single) == 1
    assert "prediction" in res_single[0]
    assert isinstance(res_single[0]["prediction"], float)
    assert res_single[0]["prediction"] > 0

    # 3. Test batch dataframe input
    df_new = pd.DataFrame([
        {"sqft": 900, "rooms": 2},
        {"sqft": 2200, "rooms": 4},
    ])
    res_batch = predict(pipeline, df_new, task_type="regression")
    assert len(res_batch) == 2
    assert res_batch[0]["prediction"] < res_batch[1]["prediction"]

    # 4. Test regression with CV metrics (confidence, margin of error, interval bounds)
    cv_metrics = {"r2_mean": 0.85, "mae_mean": 15000.0}
    res_with_metrics = predict(pipeline, raw_input, task_type="regression", metrics=cv_metrics)
    assert len(res_with_metrics) == 1
    item = res_with_metrics[0]
    assert item["confidence"] == 0.85
    assert item["confidence_metric"] == "R2 Score"
    assert item["margin_of_error"] == 15000.0
    assert item["interval_lower"] == round(item["prediction"] - 15000.0, 2)
    assert item["interval_upper"] == round(item["prediction"] + 15000.0, 2)


def test_serving_inference_regression_supports_negative_interval():
    # Model memprediksi nilai mendekati 0 atau negatif (misal temperatur)
    pipeline = Pipeline([("model", LinearRegression())])
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([-5.0, 0.0, 5.0])
    pipeline.fit(X, y)

    # Prediksi untuk X=0 menghasilkan pred=-5.0. Dengan MAE=2.0 -> interval_lower = -7.0 (negatif)
    raw_input = {"feature": 0.0}
    cv_metrics = {"r2": 0.99, "mae": 2.0}
    res = predict(pipeline, raw_input, task_type="regression", metrics=cv_metrics)

    assert res[0]["interval_lower"] < 0.0
    assert res[0]["interval_lower"] == round(res[0]["prediction"] - 2.0, 2)
