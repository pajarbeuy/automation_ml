from __future__ import annotations

# =============================================================================
# MODUL: HYPERPARAMETER TUNING ENGINE
# =============================================================================
# Deskripsi: Mengoptimalkan hiperparameter model Machine Learning menggunakan
#            pencarian acak terstruktur (RandomizedSearchCV) dan Stratified CV.
#            Menyediakan riwayat tiap trial (uji coba) dan parameter terbaik.
# =============================================================================

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from models.registry import is_regression_model


@dataclass
class TuningTrial:
    trial_index: int
    params: dict[str, Any]
    score: float
    duration_seconds: float


@dataclass
class TuningResult:
    model_name: str
    best_params: dict[str, Any]
    best_score: float
    primary_metric: str
    trials: list[dict[str, Any]]
    total_time_seconds: float


# =============================================================================
# RUANG PENCARIAN HIPERPARAMETER (SEARCH SPACES) PER ALGORITMA MODEL ML
# =============================================================================

_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    # --- Classification Models ---
    "logistic_regression": {
        "model__C": uniform(0.01, 10.0),
        "model__max_iter": [500, 1000, 1500],
        "model__solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "model__n_estimators": randint(30, 200),
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": randint(2, 10),
        "model__criterion": ["gini", "entropy"],
    },
    "gradient_boosting": {
        "model__n_estimators": randint(30, 150),
        "model__learning_rate": uniform(0.01, 0.25),
        "model__max_depth": randint(3, 10),
        "model__subsample": uniform(0.6, 0.4),
    },
    "hist_gradient_boosting": {
        "model__learning_rate": uniform(0.01, 0.25),
        "model__max_iter": randint(50, 200),
        "model__max_depth": randint(3, 15),
    },
    "knn": {
        "model__n_neighbors": randint(3, 20),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "linear_svm": {
        "model__C": uniform(0.01, 5.0),
        "model__loss": ["hinge", "squared_hinge"],
    },
    "dummy_classifier": {
        "model__strategy": ["most_frequent", "stratified", "uniform"],
    },
    # --- Regression Models ---
    "dummy_regressor": {
        "model__strategy": ["mean", "median"],
    },
    "linear_regression": {},
    "ridge": {
        "model__alpha": uniform(0.01, 100.0),
    },
    "random_forest_regressor": {
        "model__n_estimators": randint(30, 200),
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": randint(2, 10),
    },
    "gradient_boosting_regressor": {
        "model__n_estimators": randint(30, 150),
        "model__learning_rate": uniform(0.01, 0.25),
        "model__max_depth": randint(3, 10),
    },
    "hist_gradient_boosting_regressor": {
        "model__learning_rate": uniform(0.01, 0.25),
        "model__max_iter": randint(50, 200),
        "model__max_depth": randint(3, 15),
    },
    "knn_regressor": {
        "model__n_neighbors": randint(3, 20),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "xgb_regressor": {
        "model__n_estimators": randint(30, 150),
        "model__learning_rate": uniform(0.01, 0.25),
        "model__max_depth": randint(3, 10),
    },
    "lgbm_regressor": {
        "model__n_estimators": randint(30, 150),
        "model__learning_rate": uniform(0.01, 0.25),
        "model__max_depth": randint(3, 10),
    },
}


# =============================================================================
# FUNGSI EKSEKUSI HYPERPARAMETER SEARCH
# =============================================================================

def run_hyperparameter_search(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_trials: int = 10,
    random_seed: int = 42,
    primary_metric: str = "f1_weighted",
    cv_folds: int = 5,
    task_type: str = "classification",
) -> TuningResult:
    """
    Menjalankan pencarian acak hiperparameter pada Pipeline ML.

    Parameter:
        pipeline: Pipeline Scikit-Learn (preprocessor + model).
        X: Feature DataFrame.
        y: Target Series.
        model_name: Nama algoritma model ML (misal: 'random_forest').
        n_trials: Jumlah eksperimen trial kombinasi (default 10).
        random_seed: Seed reproduksibilitas.
        primary_metric: Metrik optimasi (f1, r2, mae, rmse, accuracy, roc_auc).
        cv_folds: Jumlah fold Cross-Validation.
        task_type: "classification" atau "regression".

    Return:
        TuningResult -> Hasil parameter terbaik & histori progres trial.
    """
    start_time = time.time()
    search_space = _SEARCH_SPACES.get(model_name, {})
    is_reg = task_type == "regression" or is_regression_model(model_name)

    # Pemetaan string metrik ke sklearn scorer key
    scorer_map = {
        "f1": "f1_weighted",
        "f1_weighted": "f1_weighted",
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "roc_auc": "roc_auc_ovr_weighted",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "rmse": "neg_root_mean_squared_error",
    }
    default_scorer = "r2" if is_reg else "f1_weighted"
    scoring = scorer_map.get(primary_metric, default_scorer)

    if is_reg:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    # Batasi n_trials jika ruang pencarian kecil
    actual_trials = min(n_trials, 20)

    # Eksekusi RandomizedSearchCV jika ada search space
    if search_space:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_space,
            n_iter=actual_trials,
            scoring=scoring,
            cv=cv,
            random_state=random_seed,
            n_jobs=1,
            return_train_score=False,
            refit=True,
        )
        search.fit(X, y)

        best_score = float(search.best_score_)
        # Bersihkan awalan 'model__' dari key parameter
        raw_best_params = search.best_params_
        clean_best_params = {
            k.replace("model__", ""): (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else str(v))
            for k, v in raw_best_params.items()
        }

        # Ekstraksi histori per-trial
        cv_results = search.cv_results_
        trials_list = []
        for idx in range(len(cv_results["params"])):
            p = cv_results["params"][idx]
            clean_p = {k.replace("model__", ""): str(v) for k, v in p.items()}
            score_val = float(cv_results["mean_test_score"][idx])
            time_val = float(cv_results["mean_score_time"][idx] + cv_results["mean_fit_time"][idx])
            trials_list.append({
                "trial_index": idx + 1,
                "params": clean_p,
                "score": round(score_val, 4),
                "duration_seconds": round(time_val, 3),
            })
    else:
        # Fallback jika model tidak memiliki search space (misal dummy atau linear_regression)
        pipeline.fit(X, y)
        best_score = 0.0 if is_reg else 0.5
        clean_best_params = {}
        trials_list = [{"trial_index": 1, "params": {}, "score": best_score, "duration_seconds": 0.01}]

    total_time = time.time() - start_time

    return TuningResult(
        model_name=model_name,
        best_params=clean_best_params,
        best_score=round(best_score, 4),
        primary_metric=primary_metric,
        trials=trials_list,
        total_time_seconds=round(total_time, 2),
    )
