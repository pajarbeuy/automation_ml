from __future__ import annotations

import sys
import os

# Add packages/ml_engine to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ml_engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages", "ml_engine"))

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import Base, SessionLocal, engine
import db_models  # noqa: F401 — ensure ORM models are registered
from jobs.worker import JobWorker

from routers import (
    assistant,
    auth,
    datasets,
    experiments,
    jobs,
    meta_learning,
    monitoring,
    pipelines,
    projects,
    recommendations,
    serving,
    storage,
)

# Create tables (MVP: direct DDL, no Alembic migration needed for fresh deploy)
Base.metadata.create_all(bind=engine)

def _auto_migrate_schema():
    """Add newly introduced columns to existing SQLite tables if not present."""
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            if engine.dialect.name == "sqlite":
                res = conn.execute(text("PRAGMA table_info(projects)")).fetchall()
                existing_cols = {row[1] for row in res}
                if "tenant_id" not in existing_cols:
                    conn.execute(text("ALTER TABLE projects ADD COLUMN tenant_id VARCHAR(36)"))
                if "owner_id" not in existing_cols:
                    conn.execute(text("ALTER TABLE projects ADD COLUMN owner_id VARCHAR(36)"))
                conn.commit()
        except Exception:
            pass

_auto_migrate_schema()
# Ensure storage directories exist
os.makedirs(os.path.join(settings.storage_path, "datasets"), exist_ok=True)
os.makedirs(os.path.join(settings.storage_path, "artifacts"), exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure default tenant and admin exist
    try:
        from security import get_or_create_default_user
        with SessionLocal() as db:
            get_or_create_default_user(db)
    except Exception as exc:
        pass

    worker = None
    if settings.worker_concurrency > 0:
        worker = JobWorker(
            db_session_factory=SessionLocal,
            worker_id="api-worker-main",
            poll_interval=settings.worker_poll_interval_seconds,
        )
        worker.start_in_background()
    yield
    if worker:
        worker.stop()

app = FastAPI(
    title="AutoML Human-in-the-Loop API",
    description="Assisted AutoML for tabular ML — human-controlled preprocessing and experiment workflow.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
app.include_router(monitoring.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(storage.router, prefix="/api")
app.include_router(meta_learning.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )

