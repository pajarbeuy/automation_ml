from __future__ import annotations

import sys
from pathlib import Path

# Add packages/ml_engine to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from serving.inference import predict


@pytest.fixture
def trained_pipeline():
    """Membuat pipeline klasifikasi dummy lengkap dengan Imputer, Scaler, dan Encoder."""
    df = pd.DataFrame({
        "age": [20.0, 30.0, 40.0, 50.0, 25.0],
        "fare": [10.0, 25.0, 50.0, 100.0, 15.0],
        "gender": ["male", "female", "female", "male", "female"],
        "target": [0, 1, 1, 0, 1],
    })

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, ["age", "fare"]),
        ("cat", cat_pipe, ["gender"]),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(random_state=42)),
    ])
    pipeline.fit(df[["age", "fare", "gender"]], df["target"])
    return pipeline


def test_predict_empty_inputs(trained_pipeline):
    """Memverifikasi penolakan ramah saat payload kosong (dict, list, dataframe)."""
    with pytest.raises(ValueError, match="Data input inferensi tidak boleh kosong"):
        predict(trained_pipeline, {}, task_type="classification")

    with pytest.raises(ValueError, match="Data input inferensi tidak boleh kosong"):
        predict(trained_pipeline, [], task_type="classification")

    with pytest.raises(ValueError, match="Data input inferensi tidak boleh kosong"):
        predict(trained_pipeline, pd.DataFrame(), task_type="classification")


def test_predict_missing_feature_columns(trained_pipeline):
    """Memverifikasi deteksi kolom fitur yang hilang dari input payload."""
    with pytest.raises(ValueError, match="Input inferensi tidak lengkap"):
        predict(trained_pipeline, {"age": 28.0}, task_type="classification")


def test_predict_invalid_data_type(trained_pipeline):
    """Memverifikasi pesan kesalahan yang jelas ketika tipe string dikirim ke kolom numerik."""
    with pytest.raises(ValueError, match="Tipe data tidak valid"):
        predict(
            trained_pipeline,
            {"age": "bukan_angka", "fare": 50.0, "gender": "male"},
            task_type="classification",
        )


def test_predict_unseen_categories(trained_pipeline):
    """Memverifikasi toleransi model terhadap nilai kategori baru yang belum pernah muncul saat training."""
    res = predict(
        trained_pipeline,
        {"age": 35.0, "fare": 40.0, "gender": "other"},
        task_type="classification",
    )
    assert len(res) == 1
    assert "prediction" in res[0]
    assert res[0]["prediction"] in [0, 1]
    assert "confidence" in res[0]


def test_predict_with_none_and_nan_values(trained_pipeline):
    """Memverifikasi bahwa nilai None/NaN berhasil diimputasi secara otomatis oleh pipeline."""
    res = predict(
        trained_pipeline,
        {"age": None, "fare": np.nan, "gender": None},
        task_type="classification",
    )
    assert len(res) == 1
    assert "prediction" in res[0]
    assert "confidence" in res[0]


def test_predict_extra_columns_tolerance(trained_pipeline):
    """Memverifikasi bahwa kolom ekstra yang tidak dikenal diabaikan secara aman."""
    res = predict(
        trained_pipeline,
        {"age": 29.0, "fare": 33.0, "gender": "female", "random_extra_col": 12345},
        task_type="classification",
    )
    assert len(res) == 1
    assert "prediction" in res[0]


def test_predict_dropped_columns_not_required():
    """Memverifikasi bahwa kolom yang di-drop saat training tidak wajib dikirimkan saat inferensi."""
    df = pd.DataFrame({
        "age": [20.0, 30.0, 40.0, 50.0],
        "fare": [10.0, 25.0, 50.0, 100.0],
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "street": ["Main St", "2nd St", "Oak Ave", "Pine Rd"],
        "target": [0, 1, 0, 1],
    })
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), ["age", "fare"])],
        remainder="drop",
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(random_state=42)),
    ])
    # Pipeline di-fit dengan X yang memiliki kolom date dan street
    pipeline.fit(df[["age", "fare", "date", "street"]], df["target"])

    # Saat inferensi, user HANYA mengirim age dan fare (date dan street tidak dikirim)
    res = predict(pipeline, {"age": 25.0, "fare": 30.0}, task_type="classification")
    assert len(res) == 1
    assert "prediction" in res[0]

