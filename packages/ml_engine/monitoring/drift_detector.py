from __future__ import annotations

# =============================================================================
# MODUL: MODEL & DATA DRIFT DETECTOR
# =============================================================================
# Deskripsi: Menghitung pergeseran distribusi data (covariate shift / data drift)
#            antara data training awal (baseline) dan data produksi (current).
#
# Metode Statistik:
# 1. Fitur Numerik: Kolmogorov-Smirnov 2-sample test (KS-Test) + Wasserstein Distance.
# 2. Fitur Kategorikal: Population Stability Index (PSI).
# =============================================================================

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


def detect_numeric_drift(
    baseline_series: pd.Series | np.ndarray,
    current_series: pd.Series | np.ndarray,
    feature_name: str,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Mendeteksi drift pada satu fitur numerik menggunakan uji Kolmogorov-Smirnov.

    Parameter:
        baseline_series: Data fitur numerik saat training.
        current_series: Data fitur numerik saat produksi / live inference.
        feature_name: Nama kolom fitur.
        threshold: Ambang batas p-value (default 0.05). Jika p-value < threshold, drift terdeteksi.

    Return:
        Dict hasil analisis statistik dan data binned untuk visualisasi grafik.
    """
    b_arr = np.asarray(baseline_series, dtype=float)
    c_arr = np.asarray(current_series, dtype=float)

    # Filter NaN dan inf
    b_clean = b_arr[np.isfinite(b_arr)]
    c_clean = c_arr[np.isfinite(c_arr)]

    if len(b_clean) < 3 or len(c_clean) < 3:
        return {
            "feature_name": feature_name,
            "dtype": "numeric",
            "test_name": "Kolmogorov-Smirnov Test (KS-Test)",
            "statistic": 0.0,
            "p_value": 1.0,
            "threshold": threshold,
            "wasserstein_distance": 0.0,
            "baseline_mean": round(float(np.mean(b_clean)), 4) if len(b_clean) > 0 else 0.0,
            "current_mean": round(float(np.mean(c_clean)), 4) if len(c_clean) > 0 else 0.0,
            "baseline_std": round(float(np.std(b_clean)), 4) if len(b_clean) > 0 else 0.0,
            "current_std": round(float(np.std(c_clean)), 4) if len(c_clean) > 0 else 0.0,
            "drift_detected": False,
            "status": "INSUFFICIENT_DATA",
            "distribution_chart": [],
        }

    # Jalankan KS-Test 2-sample
    ks_res = stats.ks_2samp(b_clean, c_clean)
    ks_stat = float(ks_res.statistic)
    p_val = float(ks_res.pvalue)

    # Hitung Wasserstein Distance (jarak pergeseran distribusi)
    try:
        w_dist = float(stats.wasserstein_distance(b_clean, c_clean))
    except Exception:
        w_dist = 0.0

    is_drift = bool(p_val < threshold)

    # Hitung 8 bin histogram umum untuk membandingkan distribusi
    min_val = min(float(np.min(b_clean)), float(np.min(c_clean)))
    max_val = max(float(np.max(b_clean)), float(np.max(c_clean)))

    chart_data = []
    if min_val < max_val:
        bins = np.linspace(min_val, max_val, 9)
        b_hist, _ = np.histogram(b_clean, bins=bins)
        c_hist, _ = np.histogram(c_clean, bins=bins)

        b_pct = (b_hist / len(b_clean)) * 100
        c_pct = (c_hist / len(c_clean)) * 100

        for i in range(len(bins) - 1):
            low = round(float(bins[i]), 1)
            high = round(float(bins[i + 1]), 1)
            chart_data.append({
                "bin": f"{low} - {high}",
                "baseline_pct": round(float(b_pct[i]), 2),
                "current_pct": round(float(c_pct[i]), 2),
            })

    return {
        "feature_name": feature_name,
        "dtype": "numeric",
        "test_name": "Kolmogorov-Smirnov Test (KS-Test)",
        "statistic": round(ks_stat, 4),
        "p_value": round(p_val, 4),
        "threshold": threshold,
        "wasserstein_distance": round(w_dist, 4),
        "baseline_mean": round(float(np.mean(b_clean)), 4),
        "current_mean": round(float(np.mean(c_clean)), 4),
        "baseline_std": round(float(np.std(b_clean)), 4),
        "current_std": round(float(np.std(c_clean)), 4),
        "drift_detected": is_drift,
        "status": "DRIFT" if is_drift else "STABLE",
        "distribution_chart": chart_data,
    }


def detect_categorical_drift(
    baseline_series: pd.Series | list[Any],
    current_series: pd.Series | list[Any],
    feature_name: str,
    psi_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Mendeteksi drift pada fitur kategorikal menggunakan Population Stability Index (PSI).

    Interpretasi PSI:
        - PSI < 0.1   : STABLE (Distribusi populasi stabil / tidak ada drift).
        - 0.1 <= PSI < 0.25 : MODERATE (Pergeseran moderat).
        - PSI >= 0.25 : DRIFT (Pergeseran populasi signifikan / drift terdeteksi).
    """
    b_s = pd.Series(baseline_series).dropna().astype(str)
    c_s = pd.Series(current_series).dropna().astype(str)

    if len(b_s) == 0 or len(c_s) == 0:
        return {
            "feature_name": feature_name,
            "dtype": "categorical",
            "test_name": "Population Stability Index (PSI)",
            "statistic": 0.0,
            "p_value": None,
            "threshold": psi_threshold,
            "drift_detected": False,
            "status": "INSUFFICIENT_DATA",
            "distribution_chart": [],
        }

    all_categories = sorted(list(set(b_s.unique()).union(set(c_s.unique()))))
    # Batasi ke top 12 kategori agar grafik tetap bersih
    b_val_counts = b_s.value_counts(normalize=True)
    c_val_counts = c_s.value_counts(normalize=True)

    eps = 1e-4  # Mencegah division-by-zero atau log(0)
    psi_score = 0.0
    chart_data = []

    for cat in all_categories[:12]:
        e = float(b_val_counts.get(cat, eps))
        a = float(c_val_counts.get(cat, eps))
        psi_score += (a - e) * np.log(a / e)
        chart_data.append({
            "bin": str(cat),
            "baseline_pct": round(float(b_val_counts.get(cat, 0.0)) * 100, 2),
            "current_pct": round(float(c_val_counts.get(cat, 0.0)) * 100, 2),
        })

    is_drift = bool(psi_score >= psi_threshold)
    status_label = "DRIFT" if is_drift else ("MODERATE" if psi_score >= 0.1 else "STABLE")

    return {
        "feature_name": feature_name,
        "dtype": "categorical",
        "test_name": "Population Stability Index (PSI)",
        "statistic": round(float(psi_score), 4),
        "p_value": None,
        "threshold": psi_threshold,
        "drift_detected": is_drift,
        "status": status_label,
        "distribution_chart": chart_data,
    }


def compute_dataset_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_schemas: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Menghitung metrik drift lengkap pada level dataset untuk seluruh fitur yang dipantau.

    Parameter:
        baseline_df: DataFrame dataset training asli.
        current_df: DataFrame kumpulan rekaman fitur produksi (dari PredictionLog).
        feature_schemas: Daftar spesifikasi fitur [{"name": "col", "dtype": "numeric|categorical"}].

    Return:
        Ringkasan kesehatan model, jumlah fitur drifted, dan rincian per-fitur.
    """
    feature_details: list[dict[str, Any]] = []
    drifted_count = 0
    monitored_count = 0

    for f in feature_schemas:
        col = f.get("name")
        dtype = f.get("dtype", "numeric")

        if not col or col not in baseline_df.columns or col not in current_df.columns:
            continue

        monitored_count += 1
        b_series = baseline_df[col]
        c_series = current_df[col]

        if dtype == "numeric":
            res = detect_numeric_drift(b_series, c_series, feature_name=col)
        else:
            res = detect_categorical_drift(b_series, c_series, feature_name=col)

        if res.get("drift_detected"):
            drifted_count += 1

        feature_details.append(res)

    drift_share = round((drifted_count / monitored_count), 4) if monitored_count > 0 else 0.0

    # Klasifikasi status kesehatan keseluruhan (standar industri: >= 50% fitur drifted = DRIFT_DETECTED)
    if monitored_count == 0 or len(current_df) < 5:
        overall_status = "INSUFFICIENT_DATA"
    elif drift_share >= 0.5:
        overall_status = "DRIFT_DETECTED"
    elif drift_share > 0:
        overall_status = "WARNING"
    else:
        overall_status = "HEALTHY"

    return {
        "status": overall_status,
        "total_predictions_logged": len(current_df),
        "monitored_features_count": monitored_count,
        "drifted_features_count": drifted_count,
        "drift_share": drift_share,
        "feature_details": feature_details,
    }
