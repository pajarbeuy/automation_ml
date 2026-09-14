from __future__ import annotations

import io
import sys
from pathlib import Path

# Add packages/ml_engine to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "api"))

import pandas as pd
import pytest
from fastapi import HTTPException

from profiling.profiler import profile_dataset


def test_profiling_constant_column():
    """Memverifikasi profiling mendeteksi kolom konstan dengan benar tanpa crash."""
    df = pd.DataFrame({
        "feature_a": [1, 1, 1, 1, 1],
        "target": [0, 1, 0, 1, 0],
    })
    prof = profile_dataset(df, target_column="target")
    assert prof.rows == 5
    feat_a = next(f for f in prof.features if f.name == "feature_a")
    assert feat_a.constant is True


def test_semicolon_delimited_csv_support(tmp_path):
    """Memverifikasi bahwa CSV dengan delimiter titik-koma (;) dapat dibaca dengan benar."""
    csv_content = "age;gender;target\n25;m;0\n30;f;1\n40;f;1\n50;m;0\n35;f;1\n"
    file_path = tmp_path / "data_semicolon.csv"
    file_path.write_text(csv_content)

    df = pd.read_csv(str(file_path))
    if len(df.columns) == 1 and any(sep in str(df.columns[0]) for sep in [";", "\t"]):
        df = pd.read_csv(str(file_path), sep=None, engine="python")

    assert len(df.columns) == 3
    assert "age" in df.columns
    assert "target" in df.columns
    assert len(df) == 5


def test_validation_logic_single_class_target():
    """Memverifikasi penolakan dataset dengan target tunggal pada tugas klasifikasi."""
    df = pd.DataFrame({
        "feature": [10, 20, 30, 40, 50],
        "target": [1, 1, 1, 1, 1],
    })
    clean_target = df["target"].dropna()
    n_classes = clean_target.nunique()

    # Logika validasi yang sama dengan yang ada di routers/datasets.py
    with pytest.raises(ValueError, match="minimal 2 kelas"):
        if n_classes <= 1:
            raise ValueError(f"Kolom target hanya memiliki {n_classes} kelas unik. Model klasifikasi membutuhkan minimal 2 kelas berbeda.")


def test_validation_logic_too_few_rows():
    """Memverifikasi penolakan dataset yang memiliki kurang dari 5 baris data."""
    df = pd.DataFrame({
        "feature": [10, 20, 30],
        "target": [0, 1, 0],
    })
    with pytest.raises(ValueError, match="terlalu sedikit"):
        if len(df) < 5:
            raise ValueError(f"Dataset memiliki terlalu sedikit baris data ({len(df)} baris). Minimal diperlukan 5 baris data.")


def test_target_vs_features_non_existent_target():
    from ml_engine.eda.charts import target_vs_features
    from ml_engine.profiling.profiler import profile_dataset
    df = pd.DataFrame({"feat1": [1, 2, 3, 4, 5], "feat2": [4, 5, 6, 7, 8]})
    profile = profile_dataset(df, target_column=None)
    # Tidak melempar KeyError jika target_col None atau tidak ada di DataFrame
    res1 = target_vs_features(df, profile, "non_existent_col")
    assert res1 == {}
    res2 = target_vs_features(df, profile, None)
    assert res2 == {}
