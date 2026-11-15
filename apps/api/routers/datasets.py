from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from db_models import Dataset, Project


# ML engine path resolution
_PKG = str(Path(__file__).parent.parent.parent.parent / "packages" / "ml_engine")

if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from profiling.profiler import profile_dataset
from eda.charts import build_eda
from schemas import DatasetOut, DatasetProfileOut, DatasetEDAOut

router = APIRouter(tags=["datasets"])


def _file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


@router.post(
    "/projects/{project_id}/datasets",
    response_model=DatasetOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    project_id: str,
    target_column: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    content = await file.read()
    if not content or len(content.strip()) == 0:
        raise HTTPException(status_code=400, detail="File CSV kosong. Silakan upload file CSV yang valid.")

    content_hash = _file_hash(content)

    # Persist raw CSV (immutable — never modified)
    dataset_dir = Path(settings.storage_path) / "datasets" / project_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename or "dataset.csv"
    dest = dataset_dir / filename
    with open(dest, "wb") as f:
        f.write(content)

    # Parse CSV dengan toleransi delimiter
    try:
        df = pd.read_csv(str(dest))
        if len(df.columns) == 1 and any(sep in str(df.columns[0]) for sep in [";", "\t"]):
            df = pd.read_csv(str(dest), sep=None, engine="python")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Gagal mem-parsing file CSV: {exc}")

    if df.empty or len(df) == 0:
        raise HTTPException(status_code=400, detail="File CSV tidak memiliki baris data (hanya header atau kosong).")

    if len(df) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset memiliki terlalu sedikit baris data ({len(df)} baris). Minimal diperlukan 5 baris data untuk melatih model dan cross-validation.",
        )

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found. Available: {list(df.columns)}",
        )

    clean_target = df[target_column].dropna()
    if len(clean_target) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Kolom target '{target_column}' tidak memiliki data (seluruh nilainya kosong atau NaN).",
        )

    # Validasi kelas target untuk tugas klasifikasi
    n_classes = clean_target.nunique()
    if project.task_type == "classification" and n_classes <= 1:
        raise HTTPException(
            status_code=400,
            detail=f"Kolom target '{target_column}' hanya memiliki {n_classes} kelas unik. Model klasifikasi membutuhkan minimal 2 kelas berbeda.",
        )

    try:
        profile = profile_dataset(df, target_column)
        profile_dict = profile.to_dict()

        # Run text profiling if text columns exist or project is sentiment analysis
        text_cols = [f.name for f in profile.features if f.dtype == "text"]
        if not text_cols and project.task_type in ("sentiment_analysis", "nlp"):
            str_cols = [c for c in df.columns if c != target_column and df[c].dtype == object]
            if str_cols:
                text_cols = [str_cols[0]]

        if text_cols:
            try:
                from profiling.text_profiler import profile_text
                profile_dict["text_profile"] = profile_text(df, text_cols[0], target_column)
            except Exception:
                pass

        eda_dict = build_eda(df, profile)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {exc}")

    # Update project target
    project.target_column = target_column
    project.status = "profiled"

    dataset = Dataset(
        project_id=project_id,
        filename=filename,
        storage_uri=str(dest),
        content_hash=content_hash,
        rows=len(df),
        columns=len(df.columns),
        schema_json={"columns": list(df.columns), "dtypes": {c: str(df[c].dtype) for c in df.columns}},
        profile_json=profile_dict,
        eda_json=eda_dict,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("/datasets/{dataset_id}/profile", response_model=DatasetProfileOut)
def get_profile(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/datasets/{dataset_id}/eda", response_model=DatasetEDAOut)
def get_eda(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/projects/{project_id}/datasets", response_model=list[DatasetOut])
def list_datasets(project_id: str, db: Session = Depends(get_db)):
    return db.query(Dataset).filter(Dataset.project_id == project_id).all()
