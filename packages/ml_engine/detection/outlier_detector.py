from __future__ import annotations

"""Outlier detector — analysis only, no data mutation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    """
    reports = []
    for f in profile_features:
        if f.dtype != "numeric":
            continue
        if f.outlier_rate_iqr is None or f.outlier_rate_iqr == 0:
            continue

        series = df[f.name].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_iqr = int(((series < lower) | (series > upper)).sum())

        # Robust z-score
        median = series.median()
        mad = (np.abs(series - median)).median()
        if mad > 0:
            robust_z = 0.6745 * (series - median) / mad
            n_zscore = int((np.abs(robust_z) > 3.5).sum())
            rate_zscore = n_zscore / len(series)
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
