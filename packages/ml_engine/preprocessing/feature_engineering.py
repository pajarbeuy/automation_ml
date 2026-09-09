from __future__ import annotations

"""
Automated Feature Engineering Transformer.

Applies log transformations to highly skewed numeric features and generates
interaction terms between top numeric features without data leakage.
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Automated Feature Engineering transformer for scikit-learn Pipelines.

    Parameters:
        skew_threshold: Minimum absolute skewness to trigger log1p transformation.
        create_interactions: Whether to generate interaction product features.
        max_interactions: Maximum number of interaction pairs to create.
    """

    def __init__(
        self,
        skew_threshold: float = 1.0,
        create_interactions: bool = True,
        max_interactions: int = 5,
    ) -> None:
        self.skew_threshold = skew_threshold
        self.create_interactions = create_interactions
        self.max_interactions = max_interactions

        # Fitted attributes
        self.skewed_cols_idx_: list[int] = []
        self.interaction_pairs_idx_: list[tuple[int, int]] = []
        self.n_features_in_: int = 0

    def fit(self, X: Any, y: Any = None) -> AutoFeatureEngineer:
        """Find skewed columns and top interaction pairs from training fold."""
        X_arr = np.asarray(X, dtype=float)
        self.n_features_in_ = X_arr.shape[1]

        # 1. Identify highly skewed columns with non-negative values
        self.skewed_cols_idx_ = []
        for col_idx in range(self.n_features_in_):
            col_vals = X_arr[:, col_idx]
            col_vals = col_vals[~np.isnan(col_vals)]
            if len(col_vals) > 5 and np.all(col_vals >= 0):
                skew_val = float(stats.skew(col_vals))
                if abs(skew_val) >= self.skew_threshold:
                    self.skewed_cols_idx_.append(col_idx)

        # 2. Identify top interaction feature pairs if requested
        self.interaction_pairs_idx_ = []
        if self.create_interactions and self.n_features_in_ >= 2:
            df_temp = pd.DataFrame(X_arr)
            corr_matrix = df_temp.corr().abs()
            pairs = []
            for i in range(self.n_features_in_):
                for j in range(i + 1, self.n_features_in_):
                    val = corr_matrix.iloc[i, j]
                    if not np.isnan(val):
                        pairs.append((val, i, j))
            pairs.sort(key=lambda x: x[0], reverse=True)
            self.interaction_pairs_idx_ = [(p[1], p[2]) for p in pairs[: self.max_interactions]]

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Apply log1p transformation and append interaction features."""
        X_arr = np.asarray(X, dtype=float).copy()

        # 1. Apply log1p on identified skewed columns
        for col_idx in self.skewed_cols_idx_:
            X_arr[:, col_idx] = np.log1p(np.maximum(0, X_arr[:, col_idx]))

        # 2. Append interaction features
        if self.interaction_pairs_idx_:
            interaction_cols = []
            for i, j in self.interaction_pairs_idx_:
                inter_col = X_arr[:, i] * X_arr[:, j]
                interaction_cols.append(inter_col.reshape(-1, 1))
            X_arr = np.hstack([X_arr] + interaction_cols)

        return X_arr
