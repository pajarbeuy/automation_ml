from __future__ import annotations

# =============================================================================
# ROUTER: MODEL SERVING & INFERENCE API (Updated for dropped feature handling)
# =============================================================================
# Deskripsi: Endpoint produksi untuk melayani prediksi data langsung (live inference)
#            menggunakan model kandidat terbaik Scikit-Learn Pipeline yang tersimpan.
#
# Endpoints:
# 1. GET  /api/projects/{project_id}/models/candidate/schema
# 2. POST /api/projects/{project_id}/predict
# 3. POST /api/projects/{project_id}/predict/batch-csv
# =============================================================================

import io
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Tambahkan packages/ml_engine ke sys.path jika belum ada
_ENGINE_PATH = str(Path(__file__).parent.parent.parent.parent / "packages" / "ml_engine")
if _ENGINE_PATH not in sys.path:
    sys.path.insert(0, _ENGINE_PATH)

from config import settings
from database import get_db
from db_models import Artifact, Dataset, Experiment, PredictionLog, Project
from serving.inference import load_pipeline, predict

router = APIRouter(tags=["Model Serving & Inference"])


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    features: dict[str, Any] | None = None
    instances: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Helper: Ambil Model Kandidat & Path Pipeline
# ---------------------------------------------------------------------------

def _get_candidate_and_pipeline(project_id: str, db: Session) -> tuple[Project, Experiment, str]:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project tidak ditemukan")

    candidate_exp = None
    if project.status and project.status.startswith("candidate:"):
        exp_id = project.status.split("candidate:")[1]
        candidate_exp = db.query(Experiment).filter(Experiment.id == exp_id, Experiment.status == "completed").first()

    if not candidate_exp:
        # Fallback: cari eksperimen selesai terbaru
        candidate_exp = (
            db.query(Experiment)
            .filter(Experiment.project_id == project_id, Experiment.status == "completed")
            .order_by(Experiment.finished_at.desc())
            .first()
        )

    if not candidate_exp:
        raise HTTPException(
            status_code=400,
            detail="Belum ada model kandidat yang selesai dilatih untuk project ini. Silakan latih model terlebih dahulu.",
        )

    # Cari file pipeline.pkl
    model_artifact = (
        db.query(Artifact)
        .filter(Artifact.experiment_id == candidate_exp.id, Artifact.type == "model")
        .first()
    )

    pipeline_path = None
    if model_artifact and os.path.exists(model_artifact.storage_uri):
        pipeline_path = model_artifact.storage_uri
    else:
        # Jalur cadangan filesystem
        fallback_path = os.path.join(settings.storage_path, "artifacts", candidate_exp.id, "pipeline.pkl")
        if os.path.exists(fallback_path):
            pipeline_path = fallback_path

    if not pipeline_path:
        raise HTTPException(
            status_code=404,
            detail=f"File pipeline.pkl untuk model kandidat (ID: {candidate_exp.id}) tidak ditemukan di storage.",
        )

    return project, candidate_exp, pipeline_path


# ---------------------------------------------------------------------------
# Endpoint 1: Skema Fitur Model Kandidat (Untuk Form UI Otomatis)
# ---------------------------------------------------------------------------

@router.get("/projects/{project_id}/models/candidate/schema")
def get_candidate_schema(project_id: str, db: Session = Depends(get_db)):
    """Mengembalikan daftar fitur yang diharapkan model beserta metadata tipe data dan contoh nilai."""
    project, candidate_exp, _ = _get_candidate_and_pipeline(project_id, db)

    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )
    if not dataset or not dataset.profile_json:
        raise HTTPException(status_code=400, detail="Profil dataset tidak ditemukan")

    raw_features = dataset.profile_json.get("features", [])
    dropped = (candidate_exp.config_json or {}).get("dropped_columns", [])
    target = project.target_column

    feature_schemas = []
    for f in raw_features:
        name = f.get("name")
        if name == target or name in dropped:
            continue

        dtype = "numeric" if f.get("dtype") == "numeric" else "categorical"
        example_val = f.get("mean") if dtype == "numeric" else (f.get("top_category") or "")
        categories = f.get("categories") or f.get("unique_values") or []

        feature_schemas.append({
            "name": name,
            "dtype": dtype,
            "example_value": example_val,
            "min": f.get("min"),
            "max": f.get("max"),
            "mean": f.get("mean"),
            "categories": [str(c) for c in categories[:50]] if categories else [],
        })

    return {
        "project_id": project_id,
        "candidate_experiment_id": candidate_exp.id,
        "model_name": candidate_exp.model_name,
        "task_type": project.task_type or "classification",
        "target_column": target,
        "features": feature_schemas,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helper: Asynchronous Prediction Logging
# ---------------------------------------------------------------------------

def _async_log_predictions(
    project_id: str,
    experiment_id: str,
    records: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    """Menyimpan log fitur dan hasil prediksi ke database secara non-blocking di latar belakang."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        logs = [
            PredictionLog(
                project_id=project_id,
                experiment_id=experiment_id,
                features_json=feat,
                prediction_result=res,
            )
            for feat, res in records
        ]
        db.bulk_save_objects(logs)
        db.commit()
    except Exception:
        pass
    finally:
        db.close()


def _get_candidate_cv_metrics(candidate_exp: Experiment, storage_path: str) -> dict[str, float]:
    """Mengambil metrik CV (r2_mean, mae_mean, rmse_mean) dari database atau file metrics.json."""
    metrics_map: dict[str, float] = {}
    if candidate_exp.metrics:
        for m in candidate_exp.metrics:
            if m.split == "cv":
                metrics_map[m.name] = m.value

    if "r2_mean" not in metrics_map and "mae_mean" not in metrics_map:
        metrics_file = os.path.join(storage_path, "artifacts", candidate_exp.id, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                import json
                with open(metrics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    summary = data.get("summary", {})
                    for k, v in summary.items():
                        if isinstance(v, (int, float)):
                            metrics_map[k] = float(v)
            except Exception:
                pass

    return metrics_map


# ---------------------------------------------------------------------------
# Endpoint 2: Live Inference Prediction (Single Record / Batch JSON)
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/predict")
def predict_live(
    project_id: str,
    payload: PredictRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Menjalankan inferensi live pada data fitur mentah (raw JSON).
    Menerima 'features' (single row) atau 'instances' (batch rows).
    """
    project, candidate_exp, pipeline_path = _get_candidate_and_pipeline(project_id, db)

    if payload.features is None and payload.instances is None:
        raise HTTPException(status_code=422, detail="Kirimkan 'features' (objek) atau 'instances' (array)")

    pipeline = load_pipeline(pipeline_path)
    task_type = project.task_type or "classification"
    cv_metrics = _get_candidate_cv_metrics(candidate_exp, settings.storage_path) if task_type == "regression" else None

    input_data = payload.features if payload.features is not None else payload.instances
    try:
        results = predict(pipeline, input_data, task_type=task_type, metrics=cv_metrics)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal menjalankan inferensi: {exc}")

    # Logging asinkron untuk monitoring data drift
    if payload.features is not None:
        background_tasks.add_task(
            _async_log_predictions,
            project_id=project_id,
            experiment_id=candidate_exp.id,
            records=[(payload.features, results[0])],
        )
    elif payload.instances is not None:
        background_tasks.add_task(
            _async_log_predictions,
            project_id=project_id,
            experiment_id=candidate_exp.id,
            records=list(zip(payload.instances, results)),
        )

    if payload.features is not None:
        return {
            "result": results[0],
            "model_name": candidate_exp.model_name,
            "task_type": task_type,
            "target_column": project.target_column,
        }
    else:
        return {
            "results": results,
            "count": len(results),
            "model_name": candidate_exp.model_name,
            "task_type": task_type,
            "target_column": project.target_column,
        }


# ---------------------------------------------------------------------------
# Endpoint 3: Batch CSV Prediction (Upload CSV & Download Hasil)
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/predict/batch-csv")
async def predict_batch_csv(
    project_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Menerima file CSV baru tanpa kolom target, menjalankan batch prediction,
    dan mengembalikan file CSV yang sudah dilengkapi kolom 'prediction' (dan 'confidence').
    """
    import pandas as pd

    project, candidate_exp, pipeline_path = _get_candidate_and_pipeline(project_id, db)
    task_type = project.task_type or "classification"

    content = await file.read()
    if not content or len(content.strip()) == 0:
        raise HTTPException(status_code=400, detail="File CSV kosong. Silakan upload file yang valid.")

    # Deteksi encoding UTF-8 dengan fallback Latin-1
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Gagal membaca encoding file: {exc}")

    try:
        df_input = pd.read_csv(io.StringIO(text))
        # Jika hanya 1 kolom dan ada separator lain
        if len(df_input.columns) == 1 and any(sep in df_input.columns[0] for sep in [";", "\t"]):
            df_input = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Gagal mem-parsing file CSV: {exc}")

    if df_input.empty:
        raise HTTPException(status_code=400, detail="File CSV tidak memiliki baris data (hanya header atau kosong)")

    # Hapus target column jika ada di dalam input CSV
    target = project.target_column
    if target and target in df_input.columns:
        df_features = df_input.drop(columns=[target])
    else:
        df_features = df_input.copy()

    pipeline = load_pipeline(pipeline_path)
    cv_metrics = _get_candidate_cv_metrics(candidate_exp, settings.storage_path) if task_type == "regression" else None
    try:
        preds = predict(pipeline, df_features, task_type=task_type, metrics=cv_metrics)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Gagal validasi batch CSV: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal memproses prediksi batch: {exc}")

    # Logging asinkron sampel baris batch untuk monitoring data drift (hingga 500 baris)
    sample_records = [
        (df_features.iloc[i].to_dict(), preds[i])
        for i in range(min(len(preds), 500))
    ]
    background_tasks.add_task(
        _async_log_predictions,
        project_id=project_id,
        experiment_id=candidate_exp.id,
        records=sample_records,
    )

    # Tambahkan kolom hasil ke DataFrame asli
    df_result = df_input.copy()
    df_result["prediction"] = [p["prediction"] for p in preds]

    if len(preds) > 0:
        if "confidence" in preds[0]:
            df_result["confidence"] = [p.get("confidence") for p in preds]
        if "margin_of_error" in preds[0]:
            df_result["margin_of_error"] = [p.get("margin_of_error") for p in preds]
            df_result["interval_lower"] = [p.get("interval_lower") for p in preds]
            df_result["interval_upper"] = [p.get("interval_upper") for p in preds]

    output_csv = df_result.to_csv(index=False)
    filename = f"predictions_{file.filename or 'data.csv'}"

    return Response(
        content=output_csv,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/projects/{project_id}/word-attribution")
def get_model_word_attribution(project_id: str, db: Session = Depends(get_db)):
    """Retrieve global positive and negative word attribution weights from candidate model."""
    project, candidate_exp, pipeline_path = _get_candidate_and_pipeline(project_id, db)
    try:
        pipeline = load_pipeline(pipeline_path)
        from assistant.text_explainer import extract_global_word_attributions
        attributions = extract_global_word_attributions(pipeline)
        return {
            "project_id": project_id,
            "experiment_id": candidate_exp.id,
            "model_name": candidate_exp.model_name,
            **attributions,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to extract word attributions: {exc}")
