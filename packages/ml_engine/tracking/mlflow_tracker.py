from __future__ import annotations

# =============================================================================
# MODUL: MLFLOW TRACKING INTEGRATOR
# =============================================================================
# Deskripsi: Mengintegrasikan eksperimen AutoML dengan MLflow Server / Model Registry.
#            Merekam metrik, parameter, artifak, serta objek Pipeline Scikit-Learn.
#            Dirancang Graceful Fail (tidak akan menggagalkan eksperimen jika MLflow offline).
# =============================================================================

import logging
import os
import pickle
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from experiments.runner import ExperimentRecord

logger = logging.getLogger(__name__)


# =============================================================================
# BAGIAN 1: PEMERIKSAAN KETERSEDIAAN PUSTAKA MLFLOW
# =============================================================================

def is_mlflow_available() -> bool:
    """Mengecek apakah paket `mlflow` terinstall di lingkungan Python aktif."""
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# BAGIAN 2: LOGGING EKSPERIMEN KE MLFLOW & REGISTRI MODEL
# =============================================================================

def log_experiment_to_mlflow(
    record: ExperimentRecord,
    tracking_uri: str | None = None,
    experiment_name: str = "ml-autonomus",
    register_model: bool = True,
) -> bool:
    """
    Mengirimkan seluruh metadata, metrik, parameter, dan model pipeline ke MLflow.

    Parameter:
        record: Objek ExperimentRecord dari runner.py.
        tracking_uri: URI server MLflow (misal: http://localhost:5000 atau file://./mlruns).
        experiment_name: Nama eksperimen di MLflow dashboard.
        register_model: Apakah model akan didaftarkan ke MLflow Model Registry.

    Return:
        True jika berhasil terhubung dan tercatat, False jika gagal/skip.
    """
    # 1. Pastikan mlflow terinstall
    if not is_mlflow_available():
        logger.info("mlflow is not installed. Skipping MLflow tracking.")
        return False

    import mlflow
    import mlflow.sklearn

    # Mencegah error skops security check pada pickle custom transformer
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    try:
        # 2. Atur Server Target & Nama Eksperimen
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

        is_tuned = bool(record.config_json.get("is_tuned")) if record.config_json else False
        if is_tuned and not record.model_name.endswith("_tune"):
            model_name_tag = f"{record.model_name}_tune"
        else:
            model_name_tag = record.model_name

        run_name = f"{model_name_tag}_{record.experiment_id[:8]}"
        with mlflow.start_run(run_name=run_name):
            # 3. Log Metadata & Tag
            mlflow.set_tags({
                "experiment_id": record.experiment_id,
                "project_id": record.project_id,
                "dataset_id": record.dataset_id,
                "pipeline_hash": record.pipeline_hash,
                "status": record.status,
                "is_tuned": str(is_tuned),
                "created_by": "ml-autonomus",
            })

            # 4. Log Parameter Eksperimen
            params: dict[str, Any] = {
                "model_name": model_name_tag,
                "random_seed": record.random_seed,
                "cv_folds": record.cv_folds,
                "primary_metric": record.primary_metric,
                "is_tuned": is_tuned,
            }

            if record.config_json:
                num_cfg = record.config_json.get("numeric_config", {})
                cat_cfg = record.config_json.get("categorical_config", {})
                params["numeric_columns_count"] = len(num_cfg)
                params["categorical_columns_count"] = len(cat_cfg)
                params["dropped_columns_count"] = len(record.config_json.get("dropped_columns", []))

            mlflow.log_params(params)

            # 5. Log Metrik Evaluasi (Accuracy, F1, ROC-AUC, dll)
            if record.metrics_flat:
                metrics_to_log = {}
                for k, v in record.metrics_flat.items():
                    if isinstance(v, (int, float)):
                        metrics_to_log[k] = float(v)

                if record.training_time_seconds is not None:
                    metrics_to_log["training_time_seconds"] = float(record.training_time_seconds)

                if metrics_to_log:
                    mlflow.log_metrics(metrics_to_log)

            # 6. Log Objek Pipeline ML & Daftarkan ke Model Registry
            pipeline_logged = False
            for art in record.artifacts:
                art_path = art.get("path")
                if art_path and os.path.exists(art_path):
                    if art.get("type") == "pipeline" or art_path.endswith("pipeline.pkl"):
                        try:
                            with open(art_path, "rb") as f:
                                pipeline_obj = pickle.load(f)

                            model_name_clean = model_name_tag.replace(" ", "_").lower()
                            reg_name = f"ml_autonomus_{model_name_clean}" if register_model else None

                            try:
                                mlflow.sklearn.log_model(
                                    sk_model=pipeline_obj,
                                    artifact_path="model",
                                    registered_model_name=reg_name,
                                    serialization_format="pickle",
                                )
                                pipeline_logged = True
                            except Exception as reg_err:
                                logger.info(f"Logging model without registry name due to store limits: {reg_err}")
                                mlflow.sklearn.log_model(
                                    sk_model=pipeline_obj,
                                    artifact_path="model",
                                    serialization_format="pickle",
                                )
                                pipeline_logged = True
                        except Exception as pkl_err:
                            logger.warning(f"Could not load/log pipeline object for MLflow: {pkl_err}")

                    # Upload file artifak mentah (config.json, metrics.json)
                    mlflow.log_artifact(art_path)

        logger.info(f"Successfully logged experiment {record.experiment_id} to MLflow at {uri}")
        return True

    except Exception as exc:
        logger.warning(f"Failed to log experiment to MLflow: {exc}")
        return False

