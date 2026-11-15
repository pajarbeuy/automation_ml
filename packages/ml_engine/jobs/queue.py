from __future__ import annotations

import logging
from typing import Any
from sqlalchemy.orm import Session

logger = logging.getLogger("jobs.queue")


def enqueue_job(
    db: Session,
    task_name: str,
    payload: dict[str, Any] | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    priority: int = 0,
):
    """Enqueue a new asynchronous task."""
    from db_models.orm import Job

    job = Job(
        task_name=task_name,
        payload_json=payload or {},
        project_id=project_id,
        tenant_id=tenant_id,
        priority=priority,
        status="pending",
        progress=0.0,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    logger.info(f"Enqueued job {job.id} for task {task_name}")
    return job


def get_job(db: Session, job_id: str):
    """Retrieve job by ID."""
    from db_models.orm import Job
    return db.query(Job).filter(Job.id == job_id).first()


def cancel_job(db: Session, job_id: str):
    """Cancel a pending or running job."""
    from db_models.orm import Job

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return None

    if job.status in ("completed", "failed", "cancelled"):
        return job

    job.status = "cancelled"
    job.error_message = "Job was cancelled by user request."
    db.commit()
    db.refresh(job)
    logger.info(f"Cancelled job {job.id}")
    return job


def list_project_jobs(db: Session, project_id: str):
    """List all jobs for a specific project."""
    from db_models.orm import Job
    return db.query(Job).filter(Job.project_id == project_id).order_by(Job.created_at.desc()).all()
