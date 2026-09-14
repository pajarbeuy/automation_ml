from __future__ import annotations

"""
EDA chart data generator.

All functions return JSON-serializable Python dicts/lists — no file I/O.
The frontend is responsible for rendering these data structures.
"""

from typing import Any

import numpy as np
import pandas as pd

from profiling.profiler import DatasetProfile, FeatureProfile


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def missing_value_summary(profile: DatasetProfile) -> list[dict]:
    """Bar chart data: missing rate per column, sorted descending."""
    rows = []
    for f in profile.features:
        if f.missing_count > 0:
            rows.append(
                {
                    "column": f.name,
                    "missing_count": f.missing_count,
                    "missing_rate": round(f.missing_rate * 100, 2),
                }
            )
    rows.sort(key=lambda x: x["missing_rate"], reverse=True)
    return rows


def class_distribution(profile: DatasetProfile) -> list[dict]:
    """Bar/pie chart data for target class distribution."""
    return [
        {"class": cls, "proportion": round(prop * 100, 2)}
        for cls, prop in profile.class_distribution.items()
    ]


def numerical_distributions(df: pd.DataFrame, profile: DatasetProfile) -> dict[str, dict]:
    """
    Histogram data for each numeric feature.
    Returns: {column_name: {bins: [...], counts: [...], ...}}
    """
    result = {}
    for f in profile.features:
        if f.dtype != "numeric":
            continue
        series = df[f.name].dropna()
        if len(series) < 2:
            continue
        counts, bin_edges = np.histogram(series, bins=min(30, max(5, int(len(series) ** 0.5))))
        result[f.name] = {
            "bins": [round(float(e), 4) for e in bin_edges],
            "counts": [int(c) for c in counts],
            "mean": _safe_float(f.mean),
            "median": _safe_float(f.p50),
            "std": _safe_float(f.std),
            "skewness": _safe_float(f.skewness),
            "outlier_rate_iqr": round(f.outlier_rate_iqr * 100, 2) if f.outlier_rate_iqr else 0,
        }
    return result


def boxplot_data(df: pd.DataFrame, profile: DatasetProfile) -> dict[str, dict]:
    """Box-and-whisker data for each numeric feature."""
    result = {}
    for f in profile.features:
        if f.dtype != "numeric":
            continue
        series = df[f.name].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = series[(series < lower_fence) | (series > upper_fence)].tolist()
        result[f.name] = {
            "min": _safe_float(series.min()),
            "q1": _safe_float(q1),
            "median": _safe_float(f.p50),
            "q3": _safe_float(q3),
            "max": _safe_float(series.max()),
            "lower_fence": _safe_float(lower_fence),
            "upper_fence": _safe_float(upper_fence),
            "outliers": [_safe_float(v) for v in outliers[:100]],  # cap at 100 for perf
        }
    return result


def categorical_frequencies(profile: DatasetProfile) -> dict[str, list[dict]]:
    """Top-N frequency data for each categorical feature."""
    result = {}
    for f in profile.features:
        if f.dtype not in ("categorical", "boolean"):
            continue
        result[f.name] = f.top_categories
    return result


def correlation_matrix(df: pd.DataFrame, profile: DatasetProfile) -> dict:
    """Pearson correlation matrix for numeric features."""
    numeric_cols = [f.name for f in profile.features if f.dtype == "numeric"]
    if len(numeric_cols) < 2:
        return {"columns": [], "matrix": []}
    corr = df[numeric_cols].corr(method="pearson")
    matrix = []
    for row_col in numeric_cols:
        row = []
        for col_col in numeric_cols:
            v = corr.loc[row_col, col_col]
            row.append(_safe_float(v))
        matrix.append(row)
    return {"columns": numeric_cols, "matrix": matrix}


def target_vs_features(
    df: pd.DataFrame, profile: DatasetProfile, target_col: str | None
) -> dict[str, Any]:
    """
    Feature vs target analysis.
    - Numeric features: mean per class
    - Categorical features: class proportion per category
    """
    result: dict[str, Any] = {}
    if not target_col or target_col not in df.columns:
        return result

    target = df[target_col]
    classes = sorted(target.dropna().unique().tolist(), key=str)

    for f in profile.features:
        if f.name == target_col:
            continue
        if f.dtype == "numeric":
            data = {}
            for cls in classes:
                mask = target == cls
                vals = df.loc[mask, f.name].dropna()
                data[str(cls)] = _safe_float(vals.mean()) if len(vals) > 0 else None
            result[f.name] = {"type": "numeric_mean_per_class", "data": data}

        elif f.dtype == "categorical":
            # Top 10 categories × proportion per class
            top_cats = [c["value"] for c in f.top_categories[:10]]
            data = []
            for cat in top_cats:
                row: dict[str, Any] = {"category": cat}
                for cls in classes:
                    mask = (target == cls) & (df[f.name] == cat)
                    row[str(cls)] = int(mask.sum())
                data.append(row)
            result[f.name] = {"type": "categorical_class_count", "data": data}

    return result


def build_eda(df: pd.DataFrame, profile: DatasetProfile) -> dict:
    """
    Build the complete EDA data bundle.

    Returns a single dict that the API will persist and serve as /eda.
    """
    return {
        "missing_value_summary": missing_value_summary(profile),
        "class_distribution": class_distribution(profile),
        "numerical_distributions": numerical_distributions(df, profile),
        "boxplots": boxplot_data(df, profile),
        "categorical_frequencies": categorical_frequencies(profile),
        "correlation_matrix": correlation_matrix(df, profile),
        "target_vs_features": target_vs_features(df, profile, profile.target),
    }


# =============================================================================
# BAGIAN VISUALISASI BEFORE VS AFTER SCALING (STANDARISASI & NORMALISASI)
# =============================================================================

def scaling_preview(
    df: pd.DataFrame, decisions: list[dict], profile: DatasetProfile
) -> dict[str, dict[str, Any]]:
    """
    Menghitung distribusi & statistik fitur numerik SEBELUM dan SESUDAH diterapkan Feature Scaling.

    Parameter:
        df: DataFrame dataset.
        decisions: Daftar dict WorkflowDecision ( accepted / overridden ).
        profile: Objek DatasetProfile.

    Return:
        Dict per kolom numerik berisi data histogram & statistik 'raw' vs 'transformed'.
    """
    from scipy import stats
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    from preprocessing.transformers import Winsorizer

    # Buat pencarian keputusan per kolom
    col_decisions: dict[str, dict[str, str]] = {}
    for d in decisions:
        col = d.get("column_name") or "__dataset__"
        cat = d.get("category", "")
        action = d.get("selected_action", "")
        if col not in col_decisions:
            col_decisions[col] = {}
        col_decisions[col][cat] = action

    result: dict[str, dict[str, Any]] = {}

    for f in profile.features:
        if f.dtype != "numeric" or f.name not in df.columns:
            continue

        raw_series = df[f.name].dropna()
        if len(raw_series) < 3:
            continue

        raw_vals = raw_series.values.astype(float).reshape(-1, 1)

        # Ambil aksi scaling atau outlier dari keputusan user
        dec = col_decisions.get(f.name, {})
        scaling_action = dec.get("scaling", "standard_scaling")
        outlier_action = dec.get("outlier", "leave_unchanged")

        # Tentukan objek transformer
        if outlier_action == "winsorization":
            scaler_obj = Winsorizer(factor=1.5)
            active_action = "winsorization"
        elif scaling_action == "min_max_scaling":
            scaler_obj = MinMaxScaler()
            active_action = "min_max_scaling"
        elif scaling_action == "robust_scaling":
            scaler_obj = RobustScaler()
            active_action = "robust_scaling"
        elif scaling_action in ("standard_scaling", "default"):
            scaler_obj = StandardScaler()
            active_action = "standard_scaling"
        else:
            scaler_obj = None
            active_action = "no_scaling"

        # 1. Hitung Statistik Data Mentah (Before)
        raw_counts, raw_edges = np.histogram(raw_series, bins=min(20, max(5, int(len(raw_series) ** 0.5))))
        raw_skew = float(stats.skew(raw_series)) if len(raw_series) > 2 else 0.0

        raw_stats = {
            "bins": [round(float(e), 4) for e in raw_edges],
            "counts": [int(c) for c in raw_counts],
            "mean": _safe_float(raw_series.mean()),
            "std": _safe_float(raw_series.std()),
            "min": _safe_float(raw_series.min()),
            "max": _safe_float(raw_series.max()),
            "skewness": _safe_float(raw_skew),
        }

        # 2. Hitung Transformasi & Statistik Data Sesudah (After)
        if scaler_obj is not None:
            try:
                transformed_vals = scaler_obj.fit_transform(raw_vals).ravel()
            except Exception:
                transformed_vals = raw_vals.ravel()
        else:
            transformed_vals = raw_vals.ravel()

        trans_counts, trans_edges = np.histogram(transformed_vals, bins=len(raw_counts))
        trans_skew = float(stats.skew(transformed_vals)) if len(transformed_vals) > 2 else 0.0

        transformed_stats = {
            "bins": [round(float(e), 4) for e in trans_edges],
            "counts": [int(c) for c in trans_counts],
            "mean": _safe_float(transformed_vals.mean()),
            "std": _safe_float(transformed_vals.std()),
            "min": _safe_float(transformed_vals.min()),
            "max": _safe_float(transformed_vals.max()),
            "skewness": _safe_float(trans_skew),
        }

        result[f.name] = {
            "column_name": f.name,
            "scaling_action": active_action,
            "raw": raw_stats,
            "transformed": transformed_stats,
        }

    return result

