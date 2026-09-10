from __future__ import annotations

# =============================================================================
# ROUTER: MODEL & DATA DRIFT MONITORING API
# =============================================================================
# Endpoints:
# 1. GET    /api/projects/{project_id}/drift           - Hitung & ambil laporan drift
# 2. POST   /api/projects/{project_id}/drift/simulate  - Simulasi trafik produksi (Normal vs Drifted)
# 3. DELETE /api/projects/{project_id}/drift/logs      - Bersihkan log prediksi project
# =============================================================================

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Pastikan packages/ml_engine terdaftar
_ENGINE_PATH = str(Path(__file__).parent.parent.parent.parent / "packages" / "ml_engine")
if _ENGINE_PATH not in sys.path:
    sys.path.insert(0, _ENGINE_PATH)

from config import settings
from database import get_db
from db_models import Dataset, Experiment, PredictionLog, Project
from monitoring.drift_detector import compute_dataset_drift

router = APIRouter(tags=["Model Drift Monitoring"])


class SimulateTrafficRequest(BaseModel):
    drift_type: str = Field("normal", description="'normal' atau 'drifted'")
    samples: int = Field(30, ge=5, le=200, description="Jumlah baris data yang disimulasikan")


def _get_project_and_baseline(project_id: str, db: Session) -> tuple[Project, Dataset, pd.DataFrame, list[dict[str, Any]]]:
    """Mengambil project, dataset baseline, DataFrame baseline, dan skema fitur."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project tidak ditemukan")

    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )

    dataset_path = None
    if dataset and dataset.storage_uri and os.path.exists(dataset.storage_uri):
        dataset_path = dataset.storage_uri
    elif dataset:
        rel_filename = os.path.basename(dataset.storage_uri) if dataset.storage_uri else (dataset.filename or "dataset.csv")
        possible_paths = [
            os.path.join(settings.storage_path, "datasets", project_id, rel_filename),
            os.path.join(os.path.dirname(__file__), "..", "storage", "datasets", project_id, rel_filename),
            os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api", "storage", "datasets", project_id, rel_filename),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                dataset_path = os.path.abspath(p)
                break

    if not dataset or not dataset_path:
        raise HTTPException(
            status_code=400,
            detail="File dataset training (baseline) tidak ditemukan untuk project ini.",
        )

    try:
        baseline_df = pd.read_csv(dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal membaca baseline dataset: {exc}")

    # Ambil metadata fitur dari profil dataset
    feature_schemas = []
    if dataset.profile_json and "features" in dataset.profile_json:
        target = project.target_column
        for f in dataset.profile_json["features"]:
            name = f.get("name")
            if name == target:
                continue
            dtype = "numeric" if f.get("dtype") == "numeric" else "categorical"
            feature_schemas.append({"name": name, "dtype": dtype})

    return project, dataset, baseline_df, feature_schemas


# ---------------------------------------------------------------------------
# Endpoint 1: Perhitungan & Laporan Data Drift
# ---------------------------------------------------------------------------

@router.get("/projects/{project_id}/drift")
def get_project_drift(project_id: str, db: Session = Depends(get_db)):
    """Menghitung metrik drift antara dataset training baseline dan data produksi dari PredictionLog."""
    project, dataset, baseline_df, feature_schemas = _get_project_and_baseline(project_id, db)

    # Ambil hingga 1000 log prediksi terbaru
    logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.project_id == project_id)
        .order_by(PredictionLog.created_at.desc())
        .limit(1000)
        .all()
    )

    if len(logs) < 5:
        return {
            "status": "INSUFFICIENT_DATA",
            "total_predictions_logged": len(logs),
            "min_required": 5,
            "monitored_features_count": len(feature_schemas),
            "drifted_features_count": 0,
            "drift_share": 0.0,
            "feature_details": [],
            "message": f"Log inferensi baru terkumpul {len(logs)} sampel (minimal dibutuhkan 5 sampel untuk analisis statistik). Lakukan inferensi pada Live Playground atau klik 'Simulate Traffic'.",
        }

    # Ubah log JSON menjadi DataFrame produksi
    records = [log.features_json for log in logs if isinstance(log.features_json, dict)]
    current_df = pd.DataFrame(records)

    report = compute_dataset_drift(
        baseline_df=baseline_df,
        current_df=current_df,
        feature_schemas=feature_schemas,
    )
    report["project_id"] = project_id
    report["last_prediction_at"] = logs[0].created_at.isoformat() if logs else None
    return report


# ---------------------------------------------------------------------------
# Endpoint 2: Simulasi Trafik Produksi (Normal vs Drifted)
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/drift/simulate")
def simulate_traffic(
    project_id: str,
    payload: SimulateTrafficRequest,
    db: Session = Depends(get_db),
):
    """
    Menghasilkan data inferensi sintetis ke dalam PredictionLog untuk mempermudah demo dan testing visualisasi drift.
    - 'normal': Meniru distribusi training asli dengan sedikit noise random.
    - 'drifted': Menggeser rata-rata numerik dan mengubah proporsi kategori secara ekstrem.
    """
    project, dataset, baseline_df, feature_schemas = _get_project_and_baseline(project_id, db)

    target = project.target_column
    feature_cols = [f["name"] for f in feature_schemas if f["name"] in baseline_df.columns]

    if not feature_cols:
        raise HTTPException(status_code=400, detail="Tidak ada fitur yang dapat disimulasikan.")

    sample_size = min(payload.samples, len(baseline_df))
    sample_df = baseline_df.sample(n=sample_size, replace=True, random_state=None).copy()

    new_logs = []
    for _, row in sample_df.iterrows():
        feat_dict = {}
        for f in feature_schemas:
            col = f["name"]
            if col not in row:
                continue

            val = row[col]
            if f["dtype"] == "numeric" and pd.notna(val):
                num_val = float(val)
                if payload.drift_type == "drifted":
                    # Geser nilai sebesar +80% sampai +150% untuk memicu KS-test drift
                    multiplier = np.random.uniform(1.8, 2.5)
                    feat_dict[col] = round(num_val * multiplier, 2)
                else:
                    # Noise wajar normal (+/- 5%)
                    jitter = np.random.uniform(0.95, 1.05)
                    feat_dict[col] = round(num_val * jitter, 2)
            else:
                # Kolom kategorikal
                if payload.drift_type == "drifted" and np.random.rand() > 0.4:
                    feat_dict[col] = "OUTLIER_CATEGORY"
                else:
                    feat_dict[col] = str(val) if pd.notna(val) else ""

        # Dummy result
        mock_result = {"prediction": 1 if project.task_type == "classification" else 100.0}

        new_logs.append(
            PredictionLog(
                project_id=project_id,
                experiment_id=None,
                features_json=feat_dict,
                prediction_result=mock_result,
            )
        )

    db.bulk_save_objects(new_logs)
    db.commit()

    return {
        "status": "success",
        "simulated_samples": len(new_logs),
        "drift_type": payload.drift_type,
        "message": f"Berhasil menambahkan {len(new_logs)} sampel data {payload.drift_type.upper()} ke log inferensi.",
    }


# ---------------------------------------------------------------------------
# Endpoint 3: Bersihkan Log Inferensi
# ---------------------------------------------------------------------------

@router.delete("/projects/{project_id}/drift/logs")
def clear_drift_logs(project_id: str, db: Session = Depends(get_db)):
    """Menghapus seluruh log inferensi project tertentu untuk mereset dashboard monitoring."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project tidak ditemukan")

    deleted_count = db.query(PredictionLog).filter(PredictionLog.project_id == project_id).delete()
    db.commit()

    return {
        "status": "success",
        "deleted_count": deleted_count,
        "message": f"Berhasil menghapus {deleted_count} log inferensi.",
    }
