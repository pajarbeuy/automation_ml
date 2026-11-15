from __future__ import annotations

"""
Model evaluator.

Runs stratified K-fold cross-validation and returns structured metrics.
Transformations are fitted INSIDE each fold to prevent leakage.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from models.registry import is_regression_model, supports_proba


@dataclass
class FoldResult:
    fold: int
    metrics: dict[str, float]
    train_size: int
    val_size: int
    duration_seconds: float


@dataclass
class EvaluationResult:
    model_name: str
    cv_folds: int
    fold_results: list[FoldResult]
    summary: dict[str, float]  # mean/std per metric
    confusion_matrix: list[list[int]]
    training_time_seconds: float
    residual_data: list[dict[str, float]] | None = None
    error: str | None = None


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    is_binary: bool,
    model_name: str,
) -> dict[str, float]:
    avg = "binary" if is_binary else "weighted"
    metrics: dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    if y_proba is not None:
        try:
            if is_binary:
                roc_input = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, roc_input))
                metrics["pr_auc"] = float(average_precision_score(y_true, roc_input))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
        except Exception:
            pass

    return {k: round(v, 6) for k, v in metrics.items()}


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "r2": round(r2, 6),
        "mae": round(mae, 6),
        "mse": round(mse, 6),
        "rmse": round(rmse, 6),
    }


def run_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_splits: int = 5,
    random_seed: int = 42,
    task_type: str = "classification",
    cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> EvaluationResult:
    """
    Run K-Fold (or Stratified K-Fold for classification) CV.

    Critically: the pipeline is cloned and fitted independently in each fold.
    No preprocessing information from test folds leaks into training.

    Args:
        pipeline: Unfitted sklearn Pipeline (preprocessor + model).
        X: Feature matrix (original, no prior transformations applied).
        y: Target series.
        model_name: Name for logging/tracking.
        n_splits: Number of CV folds.
        random_seed: Controls fold splitting.
        task_type: "classification" or "regression".
        cv_splits: Optional pre-generated list of (train_idx, val_idx) splits
                   to guarantee identical fold partitions across comparisons.

    Returns:
        EvaluationResult with per-fold metrics and summary stats.
    """
    from sklearn.base import clone

    is_reg = task_type == "regression" or is_regression_model(model_name)

    if cv_splits is not None:
        splits_iterator = cv_splits
        is_binary = len(sorted(y.unique())) == 2 if not is_reg else False
        can_proba = supports_proba(model_name) if not is_reg else False
        actual_splits_count = len(cv_splits)
    elif is_reg:
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits_iterator = list(cv_splitter.split(X, y))
        is_binary = False
        can_proba = False
        actual_splits_count = n_splits
    else:
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits_iterator = list(cv_splitter.split(X, y))
        classes = sorted(y.unique())
        is_binary = len(classes) == 2
        can_proba = supports_proba(model_name)
        actual_splits_count = n_splits

    fold_results: list[FoldResult] = []
    all_y_true: list = []
    all_y_pred: list = []
    total_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(splits_iterator):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_start = time.time()

        # Clone pipeline so each fold is independent
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        y_pred = fold_pipeline.predict(X_val)

        if is_reg:
            fold_metrics = _compute_regression_metrics(y_val.values, y_pred)
        else:
            y_proba = None
            if can_proba and hasattr(fold_pipeline, "predict_proba"):
                try:
                    y_proba = fold_pipeline.predict_proba(X_val)
                except Exception:
                    pass

            fold_metrics = _compute_metrics(
                y_val.values, y_pred, y_proba, is_binary, model_name
            )

        fold_duration = time.time() - fold_start

        fold_results.append(
            FoldResult(
                fold=fold_idx + 1,
                metrics=fold_metrics,
                train_size=len(X_train),
                val_size=len(X_val),
                duration_seconds=round(fold_duration, 3),
            )
        )

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())

    total_time = time.time() - total_start

    # Summary stats (mean ± std per metric)
    all_metric_keys = fold_results[0].metrics.keys() if fold_results else []
    summary: dict[str, float] = {}
    for key in all_metric_keys:
        vals = [fr.metrics[key] for fr in fold_results]
        summary[f"{key}_mean"] = round(float(np.mean(vals)), 6)
        summary[f"{key}_std"] = round(float(np.std(vals)), 6)

    residual_data: list[dict[str, float]] | None = None
    if is_reg:
        cm = []
        # Construct residual samples (capped at 500 for compact JSON payload)
        step = max(1, len(all_y_true) // 500)
        residual_data = [
            {
                "y_true": round(float(t), 4),
                "y_pred": round(float(p), 4),
                "residual": round(float(t - p), 4),
            }
            for t, p in zip(all_y_true[::step], all_y_pred[::step])
        ]
    else:
        cm = confusion_matrix(all_y_true, all_y_pred).tolist()

    return EvaluationResult(
        model_name=model_name,
        cv_folds=actual_splits_count,
        fold_results=fold_results,
        summary=summary,
        confusion_matrix=cm,
        training_time_seconds=round(total_time, 3),
        residual_data=residual_data,
    )
