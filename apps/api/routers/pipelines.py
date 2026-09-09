from __future__ import annotations

# =============================================================================
# ROUTER FASTAPI: PIPELINES (PENGELOLA PIPELINE PREPROCESSING & VALIDASI)
# =============================================================================
# Deskripsi: Endpoint API untuk menyusun konfigurasi Pipeline Scikit-Learn
#            berdasarkan keputusan preprocessing user, mendapatkan status pipeline,
#            dan melakukan dry-run (validasi pipeline tanpa melatih model).
# =============================================================================

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from db_models import Dataset, PipelineConfig, WorkflowDecision, Project


_PKG = str(Path(__file__).parent.parent.parent.parent / "packages" / "ml_engine")

if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from preprocessing.pipeline_builder import config_hash, get_pipeline_steps_description
from profiling.profiler import FeatureProfile, DatasetProfile
from schemas import PipelineConfigOut, PipelineValidationOut
from .utils import features_from_profile_json

router = APIRouter(tags=["pipelines"])


# =============================================================================
# BAGIAN 1: GENERATE PIPELINE (POST /projects/{id}/pipeline)
# =============================================================================

@router.post(
    "/projects/{project_id}/pipeline",
    response_model=PipelineConfigOut,
    status_code=status.HTTP_201_CREATED,
)
def generate_pipeline(project_id: str, db: Session = Depends(get_db)):
    """
    Mengubah seluruh keputusan preprocessing (WorkflowDecisions) yang disetujui
    menjadi konfigurasi Pipeline Scikit-Learn berstruktur JSON.
    """
    from db_models import Project

    # 1. Validasi Project
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project tidak ditemukan")

    # 2. Ambil Dataset Terakhir
    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )
    if not dataset or not dataset.profile_json:
        raise HTTPException(status_code=400, detail="Belum ada dataset yang di-profiling untuk project ini")

    # 3. Ambil Keputusan Preprocessing yang Statusnya Accepted / Overridden
    decisions = (
        db.query(WorkflowDecision)
        .filter(
            WorkflowDecision.project_id == project_id,
            WorkflowDecision.status.in_(["accepted", "overridden"]),
        )
        .all()
    )
    decisions_dicts = [
        {
            "category": d.category,
            "column_name": d.column_name,
            "selected_action": d.selected_action,
        }
        for d in decisions
    ]

    features = features_from_profile_json(dataset.profile_json)

    # 4. Rakit Konfigurasi Pipeline JSON
    from sklearn.dummy import DummyClassifier
    from preprocessing.pipeline_builder import build_pipeline
    _, config = build_pipeline(decisions_dicts, features, DummyClassifier())
    cfg_hash = config_hash(config)

    # 5. Simpan (Upsert) Konfigurasi Pipeline ke Database
    existing = (
        db.query(PipelineConfig)
        .filter(PipelineConfig.project_id == project_id)
        .first()
    )
    if existing:
        existing.config_json = config
        existing.pipeline_hash = cfg_hash
        db.commit()
        db.refresh(existing)
        return existing

    pc = PipelineConfig(
        project_id=project_id,
        config_json=config,
        pipeline_hash=cfg_hash,
    )
    db.add(pc)
    db.commit()
    db.refresh(pc)
    return pc


# =============================================================================
# BAGIAN 2: GET PIPELINE CONFIG (GET /projects/{id}/pipeline)
# =============================================================================

@router.get("/projects/{project_id}/pipeline", response_model=PipelineConfigOut)
def get_pipeline(project_id: str, db: Session = Depends(get_db)):
    """Mengambil konfigurasi pipeline aktif dari database."""
    pc = db.query(PipelineConfig).filter(PipelineConfig.project_id == project_id).first()
    if not pc:
        raise HTTPException(status_code=404, detail="Belum ada pipeline yang dikonfigurasi")
    return pc


# =============================================================================
# BAGIAN 3: VALIDASI & DRY-RUN PIPELINE (POST /projects/{id}/pipeline/validate)
# =============================================================================

@router.post("/projects/{project_id}/pipeline/validate", response_model=PipelineValidationOut)
def validate_pipeline(project_id: str, db: Session = Depends(get_db)):
    """
    Melakukan simulasi (dry-run) pembuatan pipeline tanpa melatih model.
    Memverifikasi apakah konfigurasi valid dan mengembalikan daftar langkah visual.
    """
    from db_models import Project

    import pandas as pd

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project tidak ditemukan")

    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=400, detail="Dataset tidak ditemukan")

    decisions = (
        db.query(WorkflowDecision)
        .filter(
            WorkflowDecision.project_id == project_id,
            WorkflowDecision.status.in_(["accepted", "overridden"]),
        )
        .all()
    )
    decisions_dicts = [
        {"category": d.category, "column_name": d.column_name, "selected_action": d.selected_action}
        for d in decisions
    ]
    features = features_from_profile_json(dataset.profile_json)

    warnings: list[str] = []
    errors: list[str] = []

    try:
        from sklearn.dummy import DummyClassifier
        from preprocessing.pipeline_builder import build_pipeline
        pipeline, config = build_pipeline(decisions_dicts, features, DummyClassifier())
        steps = get_pipeline_steps_description(pipeline)

        # Pemeriksaan Masalah Potensial
        if not config.get("numeric_columns") and not config.get("categorical_columns"):
            warnings.append("Tidak ada kolom fitur yang terpilih — pipeline mungkin kosong")
        if config.get("dropped_columns"):
            warnings.append(f"Kolom yang dibuang: {config['dropped_columns']}")

        return PipelineValidationOut(valid=True, steps=steps, warnings=warnings, errors=errors)

    except Exception as exc:
        errors.append(str(exc))
        return PipelineValidationOut(valid=False, steps=[], warnings=warnings, errors=errors)


# =============================================================================
# BAGIAN 4: VISUALISASI BEFORE VS AFTER SCALING (GET /projects/{id}/scaling-preview)
# =============================================================================

@router.get("/projects/{project_id}/scaling-preview")
def get_scaling_preview(project_id: str, db: Session = Depends(get_db)):
    """
    Mengembalikan data distribusi histogram & statistik sebelum dan sesudah scaling
    untuk seluruh kolom numerik berdasarkan keputusan preprocessing aktif.
    """
    import pandas as pd
    from eda.charts import scaling_preview
    from profiling.profiler import profile_dataset

    # 1. Ambil Dataset Terakhir
    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )
    if not dataset or not dataset.storage_uri:
        raise HTTPException(status_code=404, detail="Dataset tidak ditemukan")

    file_path = Path(dataset.storage_uri)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File CSV dataset tidak ditemukan pada storage")

    df = pd.read_csv(file_path)

    # 2. Ambil Profil Dataset & Target Column
    project = db.query(Project).filter(Project.id == project_id).first()
    target_col = (project.target_column if project else None) or "target"
    profile_obj = profile_dataset(df, target_column=target_col)

    # 3. Ambil Keputusan Preprocessing (Accepted / Overridden)
    decisions = (
        db.query(WorkflowDecision)
        .filter(
            WorkflowDecision.project_id == project_id,
            WorkflowDecision.status.in_(["accepted", "overridden"]),
        )
        .all()
    )
    decisions_dicts = [
        {"category": d.category, "column_name": d.column_name, "selected_action": d.selected_action}
        for d in decisions
    ]

    # 4. Hitung Preview Scaling (Before vs After)
    preview = scaling_preview(df, decisions_dicts, profile_obj)
    return preview


