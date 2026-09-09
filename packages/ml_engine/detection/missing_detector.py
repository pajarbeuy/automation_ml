from __future__ import annotations

"""Missing value detector — pure analysis, no mutation."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class MissingValueReport:
    column: str
    missing_count: int
    missing_rate: float
    dtype: str
    skewness: float | None
    outlier_rate: float | None


def detect_missing(df: pd.DataFrame, profile_features: list) -> list[MissingValueReport]:
    """
    Return missing-value reports for all columns that have at least one missing value.

    Args:
        df: Dataset (not modified).
        profile_features: List of FeatureProfile objects from profiler.
    """
    reports = []
    for f in profile_features:
        if f.missing_count == 0:
            continue
        reports.append(
            MissingValueReport(
                column=f.name,
                missing_count=f.missing_count,
                missing_rate=f.missing_rate,
                dtype=f.dtype,
                skewness=f.skewness,
                outlier_rate=f.outlier_rate_iqr,
            )
        )
    return reports
