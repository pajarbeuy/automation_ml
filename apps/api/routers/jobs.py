from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from db_models.orm import User
from jobs.queue import cancel_job, enqueue_job, get_job, list_project_jobs
from schemas.schemas import JobEnqueueIn, JobOut
from security import get_current_user, verify_project_access

router = APIRouter(prefix="/jobs", tags=["Distributed Asynchronous Jobs"])


@router.post("/enqueue", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def submit_job(
    body: JobEnqueueIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Enqueue a long-running ML task asynchronously."""
    if body.project_id:
        verify_project_access(body.project_id, db, current_user)

    job = enqueue_job(
        db=db,
        task_name=body.task_name,
        payload=body.payload,
        project_id=body.project_id,
        tenant_id=current_user.tenant_id,
        priority=body.priority,
    )
    return job


@router.get("/{job_id}", response_model=JobOut)
def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Poll job status, progress percentage, error message, and result payload."""
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.project_id:
        verify_project_access(job.project_id, db, current_user)

    return job


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_running_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Request graceful cancellation of a pending or running job."""
    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.project_id:
        verify_project_access(job.project_id, db, current_user)

    cancelled = cancel_job(db, job_id)
    return cancelled


@router.get("/project/{project_id}", response_model=list[JobOut])
def list_jobs_for_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all asynchronous background jobs associated with a project."""
    verify_project_access(project_id, db, current_user)
    return list_project_jobs(db, project_id)
