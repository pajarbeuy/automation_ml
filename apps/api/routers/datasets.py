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
    content_hash = _file_hash(content)

    # Persist raw CSV (immutable — never modified)
    dataset_dir = Path(settings.storage_path) / "datasets" / project_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename or "dataset.csv"
    dest = dataset_dir / filename
    with open(dest, "wb") as f:
        f.write(content)

    # Parse and profile
    try:
        df = pd.read_csv(str(dest))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}")

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_column}' not found. Available: {list(df.columns)}",
        )

    try:
        profile = profile_dataset(df, target_column)
        profile_dict = profile.to_dict()
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
