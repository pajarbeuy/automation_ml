from __future__ import annotations

import sys
import os

# Add packages/ml_engine to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ml_engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages", "ml_engine"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import Base, engine
import db_models  # noqa: F401 — ensure ORM models are registered

from routers import projects, datasets, recommendations, pipelines, experiments, assistant, serving

# Create tables (MVP: direct DDL, no Alembic migration needed for fresh deploy)
Base.metadata.create_all(bind=engine)

# Ensure storage directories exist
os.makedirs(os.path.join(settings.storage_path, "datasets"), exist_ok=True)
os.makedirs(os.path.join(settings.storage_path, "artifacts"), exist_ok=True)

app = FastAPI(
    title="AutoML Human-in-the-Loop API",
    description="Assisted AutoML for tabular ML — human-controlled preprocessing and experiment workflow.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(pipelines.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(assistant.router, prefix="/api")
app.include_router(serving.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
