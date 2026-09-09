from __future__ import annotations

"""
=============================================================================
MODUL PROFILING DATASET (packages/ml_engine/profiling/profiler.py)
=============================================================================
Fungsi Modul Ini:
Menganalisis dataset CSV secara otomatis tanpa mengubah data asli (read-only).
Menghitung tipe data, statistik numerik (mean, skewness, quantile), outlier,
serta distribusi kelas target untuk digunakan oleh Recommendation Engine.
=============================================================================
"""

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 1. STRUKTUR DATA PROFILING (DATACLASSES)
# =============================================================================

@dataclass
class FeatureProfile:
    """
    Menyimpan profil statistik lengkap untuk SATU kolom/fitur dalam dataset.
    """
    name: str              # Nama kolom (misal: "age", "income")
    dtype: str             # Tipe data terdeteksi: numeric / categorical / boolean / datetime / identifier
    missing_count: int     # Jumlah nilai kosong/null
    missing_rate: float    # Rasio nilai kosong (0.0 sampai 1.0)
    unique_count: int      # Jumlah nilai unik pada kolom
    constant: bool         # True jika seluruh isi kolom bernilai sama (konstan)

    # --- Statistik Khusus Kolom Numerik (Angka) ---
    mean: float | None = None          # Rata-rata
    std: float | None = None           # Standar deviasi (penyebaran data)
    min: float | None = None           # Nilai minimum
    max: float | None = None           # Nilai maksimum
    p25: float | None = None           # Kuartil 1 (25%)
    p50: float | None = None           # Median / Kuartil 2 (50%)
    p75: float | None = None           # Kuartil 3 (75%)
    skewness: float | None = None      # Kemiringan distribusi (titik berat data)
    kurtosis: float | None = None      # Keruncingan distribusi
    outlier_rate_iqr: float | None = None     # Persentase outlier berdasarkan IQR
    outlier_rate_zscore: float | None = None  # Persentase outlier berdasarkan Z-score

    # --- Statistik Khusus Kolom Kategorikal (Teks/Kategori) ---
    top_categories: list[dict] = field(default_factory=list)  # 10 kategori terbanyak & frekuensinya
    high_cardinality: bool = False                            # True jika kategori > 50 (terlalu banyak)


@dataclass
class DatasetProfile:
    """
    Menyimpan profil keseluruhan dataset (Level Dataset).
    """
    rows: int                             # Total jumlah baris data
    columns: int                          # Total jumlah kolom data
    target: str                           # Nama kolom target yang akan diprediksi
    duplicate_rows: int                   # Jumlah baris duplikat
    features: list[FeatureProfile]        # Daftar profil tiap-tiap kolom
    class_distribution: dict[str, float]  # Proporsi kelas target (misal {"0": 0.7, "1": 0.3})
    dataset_hash: str                     # Kode hash unik (SHA256) untuk verifikasi integritas dataset

    def to_dict(self) -> dict[str, Any]:
        """Mengubah objek DatasetProfile menjadi dictionary biasa."""
        return asdict(self)


# =============================================================================
# 2. HELPER KEPUTUSAN TIPE DATA & OUTLIER
# =============================================================================

def _detect_dtype(series: pd.Series) -> str:
    """
    Mendeteksi tipe data semantik dari satu kolom Pandas secara otomatis.
    Pilihan tipe: 'boolean', 'datetime', 'numeric', 'identifier', 'categorical', 'unknown'.
    """
    # 1. Cek apakah boolean (True/False atau 0/1)
    if series.dtype == bool or (
        series.dtype == object and series.dropna().isin([True, False, 0, 1, "True", "False"]).all()
    ):
        return "boolean"

    # 2. Cek apakah tipe tanggal/waktu
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # 3. Cek apakah tipe angka (numeric)
    if pd.api.types.is_numeric_dtype(series):
        # Desimal (float) pasti kolom numerik kontinyu
        if pd.api.types.is_float_dtype(series):
            return "numeric"
        # Deteksi ID unik jika nama kolom mengandung 'id'/'uuid' atau semua angka unik
        is_id_name = any(k in series.name.lower() for k in ("id", "uuid", "guid", "pk", "key", "code"))
        if is_id_name or (series.nunique() == len(series.dropna()) and series.nunique() > 100):
            return "identifier"
        return "numeric"

    # 4. Cek apakah tipe teks (string/object/categorical)
    is_string_like = (
        series.dtype == object
        or isinstance(series.dtype, pd.CategoricalDtype)
        or isinstance(series.dtype, pd.StringDtype)
        or str(series.dtype) in ("string", "str", "object")
    )

    if is_string_like:
        # Coba parse teks apakah bisa diubah jadi tanggal
        try:
            str_series = series.dropna().astype(str).head(50)
            parsed = pd.to_datetime(str_series, infer_datetime_format=True)
            if len(parsed) > 0:
                return "datetime"
        except Exception:
            pass

        # Heuristik ID unik untuk kolom string (misal: user_id, uuid, token)
        if series.nunique() == len(series.dropna()) and series.nunique() > 100:
            is_id_name = any(k in series.name.lower() for k in ("id", "uuid", "guid", "pk", "key", "code", "hash", "token"))
            if is_id_name:
                return "identifier"

        return "categorical"

    # Default fallback: jika nilai unik < 50, anggap kategorikal
    if series.nunique() < 50:
        return "categorical"

    return "unknown"


def _outlier_rate_iqr(series: pd.Series) -> float:
    """Menghitung persentase data di luar batas IQR (Interquartile Range: Q1 - 1.5*IQR s/d Q3 + 1.5*IQR)."""
    clean = series.dropna()
    if len(clean) < 4:
        return 0.0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((clean < lower) | (clean > upper)).sum()
    return float(n_outliers / len(clean))


def _outlier_rate_zscore(series: pd.Series, threshold: float = 3.0) -> float:
    """Menghitung persentase data yang memiliki deviasi ekstrem (|Z-score| > 3.0)."""
    clean = series.dropna()
    if len(clean) < 4:
        return 0.0
    std = clean.std()
    if std == 0:
        return 0.0
    z = np.abs((clean - clean.mean()) / std)
    return float((z > threshold).sum() / len(clean))


def _profile_feature(series: pd.Series, dtype: str) -> FeatureProfile:
    """Menghitung statistik lengkap untuk satu kolom tertentu."""
    n_total = len(series)
    n_missing = int(series.isna().sum())
    missing_rate = n_missing / n_total if n_total > 0 else 0.0
    unique_count = int(series.nunique())
    constant = unique_count <= 1

    fp = FeatureProfile(
        name=series.name,
        dtype=dtype,
        missing_count=n_missing,
        missing_rate=missing_rate,
        unique_count=unique_count,
        constant=constant,
    )

    # Hitung statistik numerik jika tipe data angka
    if dtype == "numeric":
        clean = series.dropna()
        if len(clean) > 0:
            fp.mean = float(clean.mean())
            fp.std = float(clean.std())
            fp.min = float(clean.min())
            fp.max = float(clean.max())
            fp.p25 = float(clean.quantile(0.25))
            fp.p50 = float(clean.quantile(0.50))
            fp.p75 = float(clean.quantile(0.75))
            fp.skewness = float(stats.skew(clean))
            fp.kurtosis = float(stats.kurtosis(clean))
            fp.outlier_rate_iqr = _outlier_rate_iqr(series)
            fp.outlier_rate_zscore = _outlier_rate_zscore(series)

    # Hitung frekuensi kategori jika tipe data teks/kategorikal
    elif dtype == "categorical":
        vc = series.value_counts(normalize=True).head(10)
        fp.top_categories = [{"value": str(k), "frequency": round(float(v), 4)} for k, v in vc.items()]
        fp.high_cardinality = unique_count > 50

    return fp


def _class_distribution(series: pd.Series) -> dict[str, float]:
    """Menghitung persentase proporsi masing-masing kelas pada kolom target."""
    vc = series.value_counts(normalize=True)
    return {str(k): round(float(v), 4) for k, v in vc.items()}


def _dataset_hash(df: pd.DataFrame) -> str:
    """Menghasilkan kode hash unik (SHA256 16-karakter) berdasarkan isi dataset."""
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


# =============================================================================
# 3. FUNGSI UTAMA PROFILING DATASET
# =============================================================================

def profile_dataset(df: pd.DataFrame, target_column: str = "") -> DatasetProfile:
    """
    Fungsi Utama: Menghitung profil lengkap dataset tanpa mengubah isi data asli.

    Args:
        df: Dataframe dataset input (sifatnya immutable / read-only).
        target_column: Nama kolom target (label yang akan diprediksi). Defaults to empty string.

    Returns:
        DatasetProfile yang berisi seluruh statistik kolom dan level dataset.
    """
    duplicate_rows = int(df.duplicated().sum())
    features: list[FeatureProfile] = []

    # Iterasi seluruh kolom (kecuali kolom target)
    for col in df.columns:
        if col == target_column:
            continue
        dtype = _detect_dtype(df[col])
        fp = _profile_feature(df[col], dtype)
        features.append(fp)

    # Profiling khusus untuk kolom target
    if target_column and target_column in df.columns:
        target_series = df[target_column]
        class_dist = _class_distribution(target_series)
    else:
        class_dist = {}

    return DatasetProfile(
        rows=len(df),
        columns=len(df.columns),
        target=target_column,
        duplicate_rows=duplicate_rows,
        features=features,
        class_distribution=class_dist,
        dataset_hash=_dataset_hash(df),
    )

