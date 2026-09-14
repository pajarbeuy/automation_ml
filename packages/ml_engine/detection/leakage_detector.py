from __future__ import annotations

"""
=============================================================================
MODUL DETEKSI KEBOCORAN DATA (detection/leakage_detector.py)
=============================================================================
Fungsi Modul Ini:
Mendeteksi potensi kebocoran data (data leakage) sebelum pelatihan model.
Kebocoran data menyebabkan model tampak sangat akurat pada pelatihan/evaluasi,
tetapi gagal total saat digunakan di produksi (generalisasi buruk).

Dua Jenis Leakage yang Dideteksi:
1. TARGET PROXY  : Fitur yang terlalu berkorelasi tinggi dengan target
                   (korelasi Pearson/Spearman > threshold, atau Cramer V > threshold).
                   Fitur ini kemungkinan "bocoran langsung" dari label target.
2. IDENTIFIER    : Kolom ID / index / nomor baris unik yang tidak membawa
                   informasi prediktif namun dapat menyebabkan overfitting total.

Prinsip:
- Read-only: Tidak mengubah DataFrame asli.
- Deterministik: Hasil konsisten untuk input yang sama.
- Evidence-driven: Setiap temuan disertai nilai statistik konkret.
=============================================================================
"""

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 1. KONFIGURASI AMBANG BATAS
# =============================================================================

# Korelasi Pearson/Spearman antara fitur numerik dan target numerik/biner
NUMERIC_TARGET_PROXY_THRESHOLD = 0.95

# Cramer V antara fitur kategorikal dan target kategorikal
CATEGORICAL_TARGET_PROXY_THRESHOLD = 0.90

# Rasio nilai unik terhadap total baris untuk mendeteksi kolom identifier
IDENTIFIER_UNIQUE_RATIO_THRESHOLD = 0.95

# Minimum jumlah baris agar korelasi bisa dihitung secara sahih
MIN_ROWS_FOR_CORRELATION = 10


# =============================================================================
# 2. STRUKTUR DATA HASIL DETEKSI
# =============================================================================

@dataclass
class LeakageReport:
    """
    Laporan potensi kebocoran data untuk SATU kolom fitur.

    Attributes:
        column: Nama kolom yang terdeteksi berisiko.
        leakage_type: Jenis leakage — 'target_proxy' atau 'identifier'.
        severity: Tingkat risiko — 'critical', 'high', atau 'medium'.
        correlation_value: Nilai statistik korelasi/asosiasi (None jika identifier).
        correlation_method: Metode yang digunakan ('pearson', 'spearman', 'cramers_v', None).
        unique_ratio: Rasio nilai unik terhadap total baris (relevan untuk identifier).
        evidence: Daftar string bukti statistik konkret.
        recommendation: Saran tindakan yang disarankan.
    """
    column: str
    leakage_type: Literal["target_proxy", "identifier"]
    severity: Literal["critical", "high", "medium"]
    correlation_value: float | None
    correlation_method: str | None
    unique_ratio: float | None
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""


# =============================================================================
# 3. HELPER PERHITUNGAN STATISTIK
# =============================================================================

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Menghitung Cramer V sebagai ukuran asosiasi antara dua variabel kategorikal.
    Nilai 0 = tidak ada asosiasi, nilai 1 = asosiasi sempurna.

    Args:
        x: Seri pertama (kategorikal).
        y: Seri kedua (kategorikal).

    Returns:
        Nilai Cramer V antara 0.0 dan 1.0.
    """
    contingency = pd.crosstab(x, y)
    n = contingency.sum().sum()
    if n == 0:
        return 0.0

    chi2, _, _, _ = stats.chi2_contingency(contingency)
    r, k = contingency.shape
    min_dim = min(r - 1, k - 1)
    if min_dim <= 0:
        return 0.0

    phi2 = chi2 / n
    v = math.sqrt(phi2 / min_dim)
    return float(min(v, 1.0))


def _point_biserial(numeric_series: pd.Series, binary_series: pd.Series) -> float:
    """
    Menghitung korelasi Point-Biserial antara fitur numerik dan target biner (0/1).

    Args:
        numeric_series: Fitur numerik kontinu.
        binary_series: Target biner (0 atau 1).

    Returns:
        Nilai absolut korelasi Point-Biserial.
    """
    combined = pd.DataFrame({"x": numeric_series, "y": binary_series}).dropna()
    if len(combined) < MIN_ROWS_FOR_CORRELATION:
        return 0.0
    corr, _ = stats.pointbiserialr(combined["x"], combined["y"])
    return float(abs(corr))


def _pearson_correlation(a: pd.Series, b: pd.Series) -> float:
    """
    Menghitung nilai absolut korelasi Pearson antara dua seri numerik.

    Returns:
        Nilai absolut korelasi Pearson (0.0 - 1.0).
    """
    combined = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(combined) < MIN_ROWS_FOR_CORRELATION:
        return 0.0
    if combined["a"].std() == 0 or combined["b"].std() == 0:
        return 0.0
    corr = combined["a"].corr(combined["b"])
    return float(abs(corr)) if not math.isnan(corr) else 0.0


def _spearman_correlation(a: pd.Series, b: pd.Series) -> float:
    """
    Menghitung nilai absolut korelasi Spearman antara dua seri (rank-based).

    Returns:
        Nilai absolut korelasi Spearman (0.0 - 1.0).
    """
    combined = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(combined) < MIN_ROWS_FOR_CORRELATION:
        return 0.0
    corr, _ = stats.spearmanr(combined["a"], combined["b"])
    return float(abs(corr)) if not math.isnan(corr) else 0.0


def _severity_from_correlation(corr: float) -> Literal["critical", "high", "medium"]:
    """Mengklasifikasikan tingkat keparahan berdasarkan nilai korelasi."""
    if corr >= 0.98:
        return "critical"
    elif corr >= 0.95:
        return "high"
    return "medium"


# =============================================================================
# 4. DETEKTOR INTERNAL
# =============================================================================

def _detect_target_proxy_numeric(
    feature_series: pd.Series,
    target_series: pd.Series,
    target_dtype: str,
    threshold: float = NUMERIC_TARGET_PROXY_THRESHOLD,
) -> tuple[float | None, str | None]:
    """
    Mendeteksi apakah kolom numerik adalah proxy dari target.

    Strategi:
    - Target numerik kontinu  -> gunakan Pearson + Spearman, ambil max
    - Target biner (0/1)      -> gunakan Point-Biserial

    Returns:
        Tuple (nilai_korelasi, metode) atau (None, None) jika di bawah threshold.
    """
    if target_dtype in ("numeric", "boolean"):
        target_clean = target_series.dropna()
        unique_vals = set(target_clean.unique())
        is_binary = unique_vals <= {0, 1, True, False, "0", "1", "True", "False"}

        if is_binary:
            target_numeric = pd.to_numeric(target_series, errors="coerce")
            corr = _point_biserial(feature_series, target_numeric)
            method = "point_biserial"
        else:
            target_numeric = pd.to_numeric(target_series, errors="coerce")
            pearson = _pearson_correlation(feature_series, target_numeric)
            spearman = _spearman_correlation(feature_series, target_numeric)
            corr = max(pearson, spearman)
            method = "pearson" if pearson >= spearman else "spearman"

        if corr >= threshold:
            return corr, method

    return None, None


def _detect_target_proxy_categorical(
    feature_series: pd.Series,
    target_series: pd.Series,
    target_dtype: str,
    threshold: float = CATEGORICAL_TARGET_PROXY_THRESHOLD,
) -> tuple[float | None, str | None]:
    """
    Mendeteksi apakah kolom kategorikal adalah proxy dari target menggunakan Cramer V.

    Returns:
        Tuple (nilai_cramers_v, 'cramers_v') atau (None, None) jika di bawah threshold.
    """
    if target_dtype in ("categorical", "boolean"):
        v = _cramers_v(feature_series.astype(str), target_series.astype(str))
        if v >= threshold:
            return v, "cramers_v"

    return None, None


def _detect_identifier_column(
    series: pd.Series,
    dtype: str,
    threshold: float = IDENTIFIER_UNIQUE_RATIO_THRESHOLD,
) -> float | None:
    """
    Mendeteksi apakah kolom adalah kolom identifier (ID, UUID, index baris).

    Pengecualian penting:
    - Kolom float kontinu (income, price, dll) TIDAK dianggap identifier
      meskipun setiap nilainya unik, karena nilai desimal kontinu memang
      cenderung unik tanpa menjadi ID.

    Returns:
        Rasio nilai unik jika kolom adalah identifier, None jika tidak.
    """
    n_total = len(series.dropna())
    if n_total == 0:
        return None

    # Kolom float kontinu bukan identifier — setiap pengukuran float memang unik
    if pd.api.types.is_float_dtype(series):
        return None

    unique_count = series.nunique()
    unique_ratio = unique_count / n_total

    # Kondisi 1: Tipe data sudah terdeteksi sebagai identifier oleh profiler
    if dtype == "identifier":
        return unique_ratio

    # Kondisi 2: Rasio keunikan sangat tinggi (kolom integer atau string)
    # Float sudah difilter di atas, jadi ini hanya berlaku untuk int/string
    if unique_ratio >= threshold and unique_count > 20:
        return unique_ratio

    return None



# =============================================================================
# 5. FUNGSI PUBLIK UTAMA
# =============================================================================

def detect_leakage(
    df: pd.DataFrame,
    target_column: str,
    feature_dtypes: dict[str, str],
    target_dtype: str | None = None,
    proxy_threshold: float | None = None,
    identifier_threshold: float = IDENTIFIER_UNIQUE_RATIO_THRESHOLD,
) -> list[LeakageReport]:
    """
    Fungsi Utama: Mendeteksi seluruh potensi kebocoran data (data leakage).

    Menjalankan dua jenis deteksi untuk setiap kolom fitur:
    1. Target Proxy: Korelasi/asosiasi yang terlalu tinggi dengan target.
    2. Identifier: Kolom yang bersifat ID unik tanpa nilai prediktif.

    Args:
        df: DataFrame dataset (read-only).
        target_column: Nama kolom target.
        feature_dtypes: Dictionary {nama_kolom: dtype_string} dari hasil profiling.
        target_dtype: Tipe data kolom target (dideteksi otomatis jika None).
        proxy_threshold: Override threshold korelasi proxy (opsional).
        identifier_threshold: Threshold rasio keunikan untuk deteksi identifier.

    Returns:
        Daftar LeakageReport diurutkan dari severity tertinggi ke terendah.
    """
    if target_column not in df.columns:
        return []

    target_series = df[target_column]

    # Deteksi tipe target secara otomatis jika tidak diberikan
    if target_dtype is None:
        if pd.api.types.is_numeric_dtype(target_series):
            target_dtype = "numeric"
        else:
            target_dtype = "categorical"

    # Tentukan threshold proxy berdasarkan tipe target
    if proxy_threshold is None:
        proxy_threshold = (
            NUMERIC_TARGET_PROXY_THRESHOLD
            if target_dtype in ("numeric", "boolean")
            else CATEGORICAL_TARGET_PROXY_THRESHOLD
        )

    reports: list[LeakageReport] = []
    severity_order = {"critical": 0, "high": 1, "medium": 2}

    for col, dtype in feature_dtypes.items():
        if col == target_column or col not in df.columns:
            continue

        feature_series = df[col]

        # ── DETEKSI 1: IDENTIFIER COLUMN ─────────────────────────────────────
        unique_ratio = _detect_identifier_column(feature_series, dtype, identifier_threshold)
        if unique_ratio is not None:
            evidence = [
                f"unique_count={feature_series.nunique()}",
                f"total_rows={len(feature_series.dropna())}",
                f"unique_ratio={round(unique_ratio * 100, 1)}%",
                f"dtype={dtype}",
            ]
            id_keywords = ("id", "uuid", "guid", "pk", "key", "code", "hash", "token", "index", "row")
            if any(kw in col.lower() for kw in id_keywords):
                evidence.append("name_contains_id_keyword=True")

            report = LeakageReport(
                column=col,
                leakage_type="identifier",
                severity="critical",
                correlation_value=None,
                correlation_method=None,
                unique_ratio=round(unique_ratio, 4),
                evidence=evidence,
                recommendation=(
                    f"Hapus kolom '{col}' sebelum melatih model. "
                    "Kolom identifier tidak membawa informasi prediktif "
                    "dan menyebabkan overfitting sempurna."
                ),
            )
            reports.append(report)
            continue  # Tidak perlu cek proxy jika sudah identifier

        # ── DETEKSI 2: TARGET PROXY ───────────────────────────────────────────
        corr_value: float | None = None
        corr_method: str | None = None

        if dtype == "numeric":
            corr_value, corr_method = _detect_target_proxy_numeric(
                feature_series=pd.to_numeric(feature_series, errors="coerce"),
                target_series=target_series,
                target_dtype=target_dtype,
                threshold=proxy_threshold,
            )

        elif dtype in ("categorical", "boolean") and target_dtype in ("categorical", "boolean"):
            corr_value, corr_method = _detect_target_proxy_categorical(
                feature_series=feature_series,
                target_series=target_series,
                target_dtype=target_dtype,
                threshold=proxy_threshold,
            )

        if corr_value is not None and corr_method is not None:
            severity = _severity_from_correlation(corr_value)
            evidence = [
                f"correlation_method={corr_method}",
                f"correlation_value={round(corr_value, 4)}",
                f"threshold={round(proxy_threshold, 2)}",
                f"dtype={dtype}",
                f"target_dtype={target_dtype}",
            ]
            report = LeakageReport(
                column=col,
                leakage_type="target_proxy",
                severity=severity,
                correlation_value=round(corr_value, 4),
                correlation_method=corr_method,
                unique_ratio=None,
                evidence=evidence,
                recommendation=(
                    f"Periksa kolom '{col}' secara manual. "
                    f"Korelasi {corr_method}={round(corr_value, 4)} terhadap target sangat tinggi. "
                    "Kemungkinan besar fitur ini mengandung informasi target yang bocor "
                    "dari proses pengumpulan data."
                ),
            )
            reports.append(report)

    # Urutkan: critical > high > medium, lalu berdasarkan nilai korelasi/ratio tertinggi
    reports.sort(
        key=lambda r: (
            severity_order.get(r.severity, 99),
            -(r.correlation_value or r.unique_ratio or 0),
        )
    )
    return reports
