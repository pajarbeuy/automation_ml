from __future__ import annotations

import sys
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

api_path = Path(__file__).parent.parent / "apps" / "api"
sys.path.insert(0, str(api_path))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))

from database import Base
from db_models.orm import Job
from jobs.queue import cancel_job, enqueue_job, get_job
from jobs.task_registry import register_task
from jobs.worker import JobWorker

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


def test_job_worker_execution_lifecycle():
    # Register a dummy task
    @register_task("test_add")
    def handle_add(payload, update_progress):
        update_progress(50.0)
        a = payload.get("a", 0)
        b = payload.get("b", 0)
        return {"sum": a + b}

    db = TestingSessionLocal()
    # 1. Enqueue job
    job = enqueue_job(db, "test_add", payload={"a": 10, "b": 32}, priority=5)
    assert job.status == "pending"
    assert job.progress == 0.0

    # 2. Worker executes single step
    worker = JobWorker(db_session_factory=TestingSessionLocal, worker_id="test-worker-1")
    executed = worker.run_once()
    assert executed is True

    # 3. Verify job finished successfully
    db.expire_all()
    updated = get_job(db, job.id)
    assert updated.status == "completed"
    assert updated.progress == 100.0
    assert updated.result_json == {"sum": 42}
    assert updated.worker_id == "test-worker-1"
    assert updated.started_at is not None
    assert updated.completed_at is not None
    db.close()


def test_job_worker_error_handling():
    @register_task("test_fail")
    def handle_fail(payload, update_progress):
        raise ValueError("Simulated task failure")

    db = TestingSessionLocal()
    job = enqueue_job(db, "test_fail", payload={})

    worker = JobWorker(db_session_factory=TestingSessionLocal, worker_id="test-worker-err")
    worker.run_once()
    db.expire_all()
    updated = get_job(db, job.id)
    assert updated.status == "failed"
    assert "Simulated task failure" in updated.error_message
    assert updated.completed_at is not None
    db.close()


def test_job_cancellation():
    db = TestingSessionLocal()
    job = enqueue_job(db, "test_add", payload={"a": 1, "b": 2})

    cancelled = cancel_job(db, job.id)
    assert cancelled.status == "cancelled"
    assert "cancelled by user request" in cancelled.error_message

    # Worker running after cancellation should skip execution
    worker = JobWorker(db_session_factory=TestingSessionLocal, worker_id="test-worker-cancel")
    executed = worker.run_once()
    assert executed is False  # No pending jobs
    db.close()
