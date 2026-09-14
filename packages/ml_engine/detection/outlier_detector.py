from __future__ import annotations

"""Outlier detector — analysis only, no data mutation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# IQR factor yang digunakan saat profiling (harus konsisten dengan profiler.py)
_IQR_FACTOR = 1.5


@dataclass
class OutlierReport:
    column: str
    n_outliers_iqr: int
    outlier_rate_iqr: float
    n_outliers_zscore: int
    outlier_rate_zscore: float
    lower_fence: float | None
    upper_fence: float | None
    skewness: float | None


def detect_outliers(df: pd.DataFrame, profile_features: list) -> list[OutlierReport]:
    """
    Return outlier reports for numeric features with outlier_rate_iqr > 0.

    The system NEVER treats outlier == error. Detection is surface-only.
    Treatment is a human decision.

    Note: outlier_rate_iqr dan lower/upper fence diturunkan dari kuartil yang
    sudah dihitung profiler (p25, p75) — tidak dihitung ulang dari DataFrame
    agar nilai tetap konsisten dengan hasil profiling yang ditampilkan ke user.
    """
    reports = []
    for f in profile_features:
        if f.dtype != "numeric":
            continue
        if f.outlier_rate_iqr is None or f.outlier_rate_iqr == 0:
            continue

        # Gunakan kuartil dari profil (sudah dihitung profiler, tidak perlu ulang)
        if f.p25 is None or f.p75 is None:
            continue

        q1, q3 = f.p25, f.p75
        iqr = q3 - q1

        # Fence dihitung dari kuartil profil agar konsisten
        lower = q1 - _IQR_FACTOR * iqr
        upper = q3 + _IQR_FACTOR * iqr

        # n_outliers_iqr diturunkan dari outlier_rate_iqr yang sudah tersimpan
        series = df[f.name].dropna()
        n_total = len(series)
        n_iqr = round(f.outlier_rate_iqr * n_total)

        # Robust z-score — dihitung dari df karena profil tidak menyimpan nilai ini
        median = series.median()
        mad = (np.abs(series - median)).median()
        if mad > 0:
            robust_z = 0.6745 * (series - median) / mad
            n_zscore = int((np.abs(robust_z) > 3.5).sum())
            rate_zscore = n_zscore / n_total
        else:
            n_zscore = 0
            rate_zscore = 0.0

        reports.append(
            OutlierReport(
                column=f.name,
                n_outliers_iqr=n_iqr,
                outlier_rate_iqr=f.outlier_rate_iqr,
                n_outliers_zscore=n_zscore,
                outlier_rate_zscore=rate_zscore,
                lower_fence=float(lower),
                upper_fence=float(upper),
                skewness=f.skewness,
            )
        )
    return reports
