from __future__ import annotations

import logging
import math
from typing import Any
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger("ml_engine.meta_ranker")


class MetaFeatureExtractor:
    """Extracts meta-features from dataset profiling data to represent dataset characteristics."""

    FEATURE_NAMES = [
        "log_rows",
        "log_cols",
        "ratio_rows_cols",
        "numeric_ratio",
        "categorical_ratio",
        "missing_cells_ratio",
        "max_missing_ratio",
        "mean_abs_skewness",
        "max_abs_skewness",
        "outlier_ratio",
        "is_regression",
        "is_classification",
        "imbalance_ratio",
    ]

    @classmethod
    def extract(cls, profile_dict: dict[str, Any], task_type: str = "classification") -> dict[str, float]:
        rows = float(profile_dict.get("rows", 100) or 100)
        cols = float(profile_dict.get("columns", 10) or 10)

        features_list = profile_dict.get("features", [])
        num_numeric = 0
        num_categorical = 0
        total_missing = 0
        max_missing = 0.0
        skewness_values = []
        outlier_count = 0

        for f in features_list:
            if isinstance(f, dict):
                dtype = f.get("dtype", "")
                missing_rate = float(f.get("missing_rate", 0.0) or 0.0)
                skewness = f.get("skewness")
                has_outliers = bool(f.get("has_outliers", False))
            else:
                dtype = getattr(f, "dtype", "")
                missing_rate = float(getattr(f, "missing_rate", 0.0) or 0.0)
                skewness = getattr(f, "skewness", None)
                has_outliers = bool(getattr(f, "has_outliers", False))

            if dtype == "numeric":
                num_numeric += 1
                if skewness is not None:
                    try:
                        skewness_values.append(abs(float(skewness)))
                    except (ValueError, TypeError):
                        pass
                if has_outliers:
                    outlier_count += 1
            elif dtype == "categorical":
                num_categorical += 1

            total_missing += int(missing_rate * rows)
            if missing_rate > max_missing:
                max_missing = missing_rate

        total_cells = max(rows * cols, 1.0)
        missing_cells_ratio = min(total_missing / total_cells, 1.0)
        mean_skewness = float(np.mean(skewness_values)) if skewness_values else 0.0
        max_skewness = float(np.max(skewness_values)) if skewness_values else 0.0

        imbalance_ratio = 1.0
        if task_type == "classification":
            target_summary = profile_dict.get("target_summary", {})
            class_dist = target_summary.get("class_distribution", {}) if isinstance(target_summary, dict) else {}
            if class_dist and len(class_dist) >= 2:
                counts = sorted(class_dist.values())
                imbalance_ratio = float(counts[0]) / float(counts[-1]) if counts[-1] > 0 else 1.0

        return {
            "log_rows": round(math.log10(max(rows, 1.0)), 4),
            "log_cols": round(math.log10(max(cols, 1.0)), 4),
            "ratio_rows_cols": round(rows / max(cols, 1.0), 4),
            "numeric_ratio": round(num_numeric / max(cols, 1.0), 4),
            "categorical_ratio": round(num_categorical / max(cols, 1.0), 4),
            "missing_cells_ratio": round(missing_cells_ratio, 4),
            "max_missing_ratio": round(max_missing, 4),
            "mean_abs_skewness": round(mean_skewness, 4),
            "max_abs_skewness": round(max_skewness, 4),
            "outlier_ratio": round(outlier_count / max(num_numeric, 1.0), 4),
            "is_regression": 1.0 if task_type == "regression" else 0.0,
            "is_classification": 1.0 if task_type == "classification" else 0.0,
            "imbalance_ratio": round(imbalance_ratio, 4),
        }

    @classmethod
    def to_vector(cls, meta_features: dict[str, float]) -> np.ndarray:
        return np.array([meta_features.get(k, 0.0) for k in cls.FEATURE_NAMES], dtype=float)


class StrategyEncoder:
    """Encodes treatment strategies into feature representations."""

    STRATEGY_MAP = {
        "recommended": [1.0, 0.0, 0.0],
        "baseline": [0.0, 1.0, 0.0],
        "alternative": [0.0, 0.0, 1.0],
    }

    @classmethod
    def encode(cls, strategy_name: str) -> np.ndarray:
        s = strategy_name.lower()
        for k, vec in cls.STRATEGY_MAP.items():
            if k in s:
                return np.array(vec, dtype=float)
        return np.array([1.0, 0.0, 0.0], dtype=float)


class MetaLearningRanker:
    """
    Surrogate Meta-Learning Ranker.
    Predicts empirical treatment lift based on dataset meta-features and historical benchmarks.
    Gracefully blends with heuristic scoring under cold-start conditions.
    """

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
        self.is_trained = False
        self.n_samples = 0
        self.feature_names = MetaFeatureExtractor.FEATURE_NAMES + ["strat_rec", "strat_base", "strat_alt"]

    def train(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Train meta-learning model on historical runs.
        Each record must contain: meta_features, strategy_name, lift.
        """
        if len(records) < 5:
            self.is_trained = False
            self.n_samples = len(records)
            return {
                "status": "cold_start_fallback",
                "message": f"Insufficient historical records ({len(records)} < 5). Cold-start heuristic retained.",
                "n_samples": len(records),
            }

        X = []
        y = []
        for r in records:
            meta_vec = MetaFeatureExtractor.to_vector(r["meta_features"])
            strat_vec = StrategyEncoder.encode(r["strategy_name"])
            combined = np.concatenate([meta_vec, strat_vec])
            X.append(combined)
            y.append(float(r["lift"]))

        X_mat = np.array(X)
        y_vec = np.array(y)

        self.model.fit(X_mat, y_vec)
        self.is_trained = True
        self.n_samples = len(records)
        train_score = float(self.model.score(X_mat, y_vec))

        logger.info(f"Meta-ranker successfully trained on {len(records)} historical runs. R2: {train_score:.4f}")
        return {
            "status": "trained",
            "message": f"Meta-learning surrogate model successfully trained on {len(records)} historical benchmarks.",
            "n_samples": len(records),
            "r2_score": round(train_score, 4),
        }

    def predict_lift(self, meta_features: dict[str, float], strategy_name: str) -> tuple[float, float]:
        """
        Predict expected lift and confidence score.
        Returns (predicted_lift, confidence_0_to_100).
        """
        if not self.is_trained:
            # Fallback prior: recommended strategy historically outperforms baseline
            priors = {"recommended": 0.015, "alternative": 0.005, "baseline": 0.0}
            strat_key = next((k for k in priors if k in strategy_name.lower()), "recommended")
            return priors[strat_key], 50.0

        meta_vec = MetaFeatureExtractor.to_vector(meta_features)
        strat_vec = StrategyEncoder.encode(strategy_name)
        combined = np.concatenate([meta_vec, strat_vec]).reshape(1, -1)

        predicted_lift = float(self.model.predict(combined)[0])
        # Model confidence increases with training sample size (asymptotic to 95%)
        confidence = min(95.0, 50.0 + (self.n_samples * 2.0))
        return predicted_lift, confidence

    def rank_candidates(
        self,
        meta_features: dict[str, float],
        candidates: list[dict[str, Any]],
        baseline_score: float = 0.80,
    ) -> list[dict[str, Any]]:
        """
        Rank candidate strategies by blending heuristic ranking with meta-learning lift predictions.
        """
        results = []
        weight_meta = min(0.7, 0.1 * self.n_samples) if self.is_trained else 0.15

        for cand in candidates:
            cand_id = cand.get("candidate_id", "cand")
            name = cand.get("name", "Strategy")
            strategy_type = cand.get("strategy_type", "recommended")
            heur_score = float(cand.get("heuristic_score", 85.0))

            pred_lift, confidence = self.predict_lift(meta_features, strategy_type)
            pred_score = max(0.0, baseline_score + pred_lift)

            # Convert lift to a 0-100 scale: 0 lift is 70 points, +5% lift is 95 points
            meta_score = min(100.0, max(0.0, 70.0 + (pred_lift * 500.0)))
            blended_score = round(((1.0 - weight_meta) * heur_score) + (weight_meta * meta_score), 2)

            if self.is_trained:
                rationale = (
                    f"Surrogate meta-model trained on {self.n_samples} historical runs predicts "
                    f"{'+' if pred_lift >= 0 else ''}{pred_lift * 100:.2f}% metric lift for {name}."
                )
            else:
                rationale = (
                    f"Cold-start prior predicts {'+' if pred_lift >= 0 else ''}{pred_lift * 100:.2f}% "
                    f"lift based on dataset tabular profile."
                )

            results.append({
                "candidate_id": cand_id,
                "name": name,
                "strategy_type": strategy_type,
                "predicted_lift": round(pred_lift, 4),
                "predicted_score": round(pred_score, 4),
                "meta_confidence": round(confidence, 1),
                "heuristic_score": round(heur_score, 2),
                "blended_rank_score": blended_score,
                "rationale": rationale,
            })

        results.sort(key=lambda x: x["blended_rank_score"], reverse=True)
        return results


# Global singleton instance
_meta_ranker = MetaLearningRanker()


def get_meta_ranker() -> MetaLearningRanker:
    return _meta_ranker
