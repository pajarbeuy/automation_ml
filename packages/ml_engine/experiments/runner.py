from __future__ import annotations

# =============================================================================
# MODUL: EXPERIMENT RUNNER (PENGESEKUSI EKSPERIMEN MODEL ML)
# =============================================================================
# Deskripsi: Mengatur alur kerja pelatihan dan evaluasi model Machine Learning:
#            Muat Data -> Bangun Pipeline Preprocessing -> Cross Validation -> Simpan Artifak & MLflow.
#            Menangani error secara modular sehingga tidak menyebabkan server crash.
# =============================================================================

import hashlib
import json
import os
import pickle
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

from evaluation.evaluator import EvaluationResult, run_cross_validation
from models.registry import get_estimator, supports_proba
from preprocessing.pipeline_builder import build_pipeline, config_hash


# =============================================================================
# BAGIAN 1: STRUKTUR DATA EKSPERIMEN (EXPERIMENT RECORD DATACLASS)
# =============================================================================

@dataclass
class ExperimentRecord:
    """
    Data class yang menyimpan seluruh informasi hasil eksperimen pelatihan model,
    siap untuk disimpan ke database SQLite / PostgreSQL dan MLflow Tracking.
    """
    experiment_id: str
    project_id: str
    dataset_id: str
    model_name: str
    random_seed: int
    cv_folds: int
    primary_metric: str
    config_json: dict
    pipeline_hash: str
    status: str  # 'completed' atau 'failed'
    started_at: str
    finished_at: str
    training_time_seconds: float | None
    error_message: str | None
    metrics_flat: dict[str, float]  # Ringkasan metrik (Accuracy, F1, R2, MAE, dll)
    fold_metrics: list[dict]       # Hasil rincian metrik per fold CV
    confusion_matrix: list[list[int]] # Matriks Kebingungan (Confusion Matrix)
    artifacts: list[dict]          # Daftar path artifak (config.json, metrics.json, pipeline.pkl)
    environment: dict              # Metadata lingkungan (versi Python, Sklearn, dll)
    residual_data: list[dict] | None = None


# =============================================================================
# BAGIAN 2: FUNGSI PEMBANTU UTILITY
# =============================================================================

def _file_checksum(path: str) -> str:
    """Menghitung checksum SHA256 file artifak untuk verifikasi integritas data."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _environment_snapshot() -> dict:
    """Mengambil snapshot versi pustaka utama untuk merekam jejak reproduksibilitas."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
    }


# =============================================================================
# BAGIAN 3: FUNGSI UTAMA EKSEKUSI EKSPERIMEN (RUN_EXPERIMENT)
# =============================================================================

def run_experiment(
    experiment_id: str,
    project_id: str,
    dataset_id: str,
    dataset_path: str,
    target_column: str,
    model_name: str,
    decisions: list[dict],
    profile_features: list,
    storage_path: str,
    random_seed: int = 42,
    cv_folds: int = 5,
    primary_metric: str = "f1",
    enable_feature_engineering: bool = False,
    task_type: str = "classification",
    model_params: dict[str, Any] | None = None,
) -> ExperimentRecord:
    """
    Menjalankan 1 eksperimen pelatihan model Machine Learning dari awal hingga selesai.

    Tahapan:
        1. Memuat CSV dataset.
        2. Membangun Pipeline Scikit-Learn dari keputusan preprocessing & hyperparameter tuning.
        3. Menjalankan N-Fold Cross Validation.
        4. Menyimpan file artifak (konfigurasi, metrik, model pickle).
        5. Mengirimkan catatan eksperimen ke MLflow Tracking Server.

    Return:
        ExperimentRecord -> Objek hasil lengkap eksperimen.
    """
    started_at = datetime.utcnow().isoformat()
    artifacts: list[dict] = []
    error_message: str | None = None

    # -------------------------------------------------------------------------
    # TAHAP 1: Memuat Data CSV
    # -------------------------------------------------------------------------
    try:
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        return _failed_record(
            experiment_id, project_id, dataset_id, model_name, random_seed,
            cv_folds, primary_metric, {}, "", started_at,
            f"LOAD_ERROR: {exc}", artifacts,
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # -------------------------------------------------------------------------
    # TAHAP 2: Membangun Pipeline Preprocessing & Estimator Model
    # -------------------------------------------------------------------------
    try:
        overrides = dict(model_params) if model_params else {}
        estimator = get_estimator(model_name, random_seed=random_seed, task_type=task_type, **overrides)
        pipeline, cfg = build_pipeline(
            decisions,
            profile_features,
            estimator,
            enable_feature_engineering=enable_feature_engineering,
        )
        if model_params:
            cfg["is_tuned"] = True
            cfg["model_params"] = model_params
        cfg_hash = config_hash(cfg)
    except Exception as exc:
        return _failed_record(
            experiment_id, project_id, dataset_id, model_name, random_seed,
            cv_folds, primary_metric, {}, "", started_at,
            f"PIPELINE_BUILD_ERROR: {exc}", artifacts,
        )

    # -------------------------------------------------------------------------
    # TAHAP 3: Menjalankan N-Fold Cross Validation
    # -------------------------------------------------------------------------
    try:
        result: EvaluationResult = run_cross_validation(
            pipeline=pipeline,
            X=X,
            y=y,
            model_name=model_name,
            n_splits=cv_folds,
            random_seed=random_seed,
            task_type=task_type,
        )
    except Exception as exc:
        return _failed_record(
            experiment_id, project_id, dataset_id, model_name, random_seed,
            cv_folds, primary_metric, cfg, cfg_hash, started_at,
            f"TRAINING_ERROR: {exc}", artifacts,
        )

    # -------------------------------------------------------------------------
    # TAHAP 4: Menyimpan Artifak Hasil Eksperimen
    # -------------------------------------------------------------------------
    artifact_dir = Path(storage_path) / "artifacts" / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # A. Simpan JSON Konfigurasi Pipeline
    cfg_path = str(artifact_dir / "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    artifacts.append({"type": "config", "path": cfg_path, "checksum": _file_checksum(cfg_path)})

    # B. Simpan JSON Metrik Evaluasi, Confusion Matrix & Residual Data
    metrics_path = str(artifact_dir / "metrics.json")
    metrics_data = {
        "summary": result.summary,
        "fold_results": [asdict(fr) for fr in result.fold_results],
        "confusion_matrix": result.confusion_matrix,
        "residual_data": result.residual_data,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    artifacts.append({"type": "metrics", "path": metrics_path, "checksum": _file_checksum(metrics_path)})

    # C. Fit Pipeline pada Seluruh Data Train & Simpan Objek Model (.pkl)
    try:
        from sklearn.base import clone
        final_pipeline = clone(pipeline)
        final_pipeline.fit(X, y)

        if task_type in ("sentiment_analysis", "nlp"):
            try:
                from assistant.text_explainer import extract_global_word_attributions
                metrics_data["word_attributions"] = extract_global_word_attributions(final_pipeline)
                with open(metrics_path, "w") as f:
                    json.dump(metrics_data, f, indent=2)
            except Exception:
                pass

        model_path = str(artifact_dir / "pipeline.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(final_pipeline, f)
        artifacts.append({"type": "pipeline", "path": model_path, "checksum": _file_checksum(model_path)})
    except Exception:
        pass  # Non-fatal error jika penyimpanan pickle gagal

    finished_at = datetime.utcnow().isoformat()
    fold_metrics = [asdict(fr) for fr in result.fold_results]

    record = ExperimentRecord(
        experiment_id=experiment_id,
        project_id=project_id,
        dataset_id=dataset_id,
        model_name=model_name,
        random_seed=random_seed,
        cv_folds=cv_folds,
        primary_metric=primary_metric,
        config_json=cfg,
        pipeline_hash=cfg_hash,
        status="completed",
        started_at=started_at,
        finished_at=finished_at,
        training_time_seconds=result.training_time_seconds,
        error_message=None,
        metrics_flat=result.summary,
        fold_metrics=fold_metrics,
        confusion_matrix=result.confusion_matrix,
        artifacts=artifacts,
        environment=_environment_snapshot(),
        residual_data=result.residual_data,
    )

    # -------------------------------------------------------------------------
    # TAHAP 5: Registrasi / Tracking ke MLflow Server
    # -------------------------------------------------------------------------
    try:
        from tracking import log_experiment_to_mlflow
        log_experiment_to_mlflow(record)
    except Exception:
        pass

    return record


# =============================================================================
# BAGIAN 4: FUNGSI CATATAN EKSPERIMEN GAGAL
# =============================================================================

def _failed_record(
    experiment_id, project_id, dataset_id, model_name, random_seed,
    cv_folds, primary_metric, config_json, pipeline_hash, started_at,
    error_message, artifacts,
) -> ExperimentRecord:
    """Membangun objek ExperimentRecord berstatus 'failed' ketika terjadi kegagalan."""
    return ExperimentRecord(
        experiment_id=experiment_id,
        project_id=project_id,
        dataset_id=dataset_id,
        model_name=model_name,
        random_seed=random_seed,
        cv_folds=cv_folds,
        primary_metric=primary_metric,
        config_json=config_json,
        pipeline_hash=pipeline_hash,
        status="failed",
        started_at=started_at,
        finished_at=datetime.utcnow().isoformat(),
        training_time_seconds=None,
        error_message=error_message,
        metrics_flat={},
        fold_metrics=[],
        confusion_matrix=[],
        artifacts=artifacts,
        environment=_environment_snapshot(),
    )

