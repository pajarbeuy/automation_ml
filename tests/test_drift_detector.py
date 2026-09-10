from __future__ import annotations

import sys
from pathlib import Path

# Add packages/ml_engine to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))

import numpy as np
import pandas as pd
import pytest

from monitoring.drift_detector import (
    compute_dataset_drift,
    detect_categorical_drift,
    detect_numeric_drift,
)


def test_detect_numeric_drift_stable():
    """Memverifikasi KS-Test mengenali data numerik yang berasal dari populasi identik (STABLE)."""
    np.random.seed(42)
    baseline = np.random.normal(loc=100.0, scale=15.0, size=150)
    current = np.random.normal(loc=100.0, scale=15.0, size=150)

    res = detect_numeric_drift(baseline, current, feature_name="salary")
    assert res["feature_name"] == "salary"
    assert res["dtype"] == "numeric"
    assert res["drift_detected"] is False
    assert res["status"] == "STABLE"
    assert res["p_value"] > 0.05
    assert len(res["distribution_chart"]) > 0


def test_detect_numeric_drift_significant():
    """Memverifikasi KS-Test mendeteksi pergeseran rata-rata ekstrim (DRIFT)."""
    np.random.seed(42)
    baseline = np.random.normal(loc=100.0, scale=10.0, size=100)
    current = np.random.normal(loc=250.0, scale=10.0, size=100)

    res = detect_numeric_drift(baseline, current, feature_name="salary")
    assert res["drift_detected"] is True
    assert res["status"] == "DRIFT"
    assert res["p_value"] < 0.05
    assert res["wasserstein_distance"] > 100.0


def test_detect_categorical_drift_stable():
    """Memverifikasi bahwa Population Stability Index (PSI) < 0.1 saat distribusi kategori seimbang."""
    baseline = pd.Series(["A", "A", "B", "B", "C"] * 20)
    current = pd.Series(["A", "A", "B", "B", "C"] * 20)

    res = detect_categorical_drift(baseline, current, feature_name="tier")
    assert res["drift_detected"] is False
    assert res["status"] == "STABLE"
    assert res["statistic"] < 0.1
    assert len(res["distribution_chart"]) > 0


def test_detect_categorical_drift_significant():
    """Memverifikasi PSI >= 0.25 saat distribusi kategori bergeser drastis."""
    baseline = pd.Series(["Gold"] * 80 + ["Silver"] * 15 + ["Bronze"] * 5)
    current = pd.Series(["Bronze"] * 80 + ["Silver"] * 15 + ["Gold"] * 5)

    res = detect_categorical_drift(baseline, current, feature_name="tier")
    assert res["drift_detected"] is True
    assert res["status"] == "DRIFT"
    assert res["statistic"] >= 0.25


def test_compute_dataset_drift_summary():
    """Memverifikasi kalkulasi agregat level dataset dengan kombinasi kolom stabil dan drift."""
    np.random.seed(42)
    baseline_df = pd.DataFrame({
        "age": np.random.normal(30, 5, 100),
        "income": np.random.normal(5000, 500, 100),
        "dept": ["IT"] * 50 + ["HR"] * 50,
    })

    # income digeser, age & dept stabil
    current_df = pd.DataFrame({
        "age": np.random.normal(30, 5, 50),
        "income": np.random.normal(25000, 500, 50),
        "dept": ["IT"] * 25 + ["HR"] * 25,
    })

    schemas = [
        {"name": "age", "dtype": "numeric"},
        {"name": "income", "dtype": "numeric"},
        {"name": "dept", "dtype": "categorical"},
    ]

    report = compute_dataset_drift(baseline_df, current_df, schemas)
    assert report["total_predictions_logged"] == 50
    assert report["monitored_features_count"] == 3
    assert report["drifted_features_count"] == 1  # hanya income yang drift
    assert report["status"] == "WARNING"
    assert len(report["feature_details"]) == 3


def test_drift_insufficient_samples():
    """Memverifikasi penanganan aman ketika sampel produksi < 3."""
    baseline = np.array([10.0, 20.0, 30.0, 40.0])
    current = np.array([15.0])

    res = detect_numeric_drift(baseline, current, feature_name="age")
    assert res["status"] == "INSUFFICIENT_DATA"
    assert res["drift_detected"] is False
