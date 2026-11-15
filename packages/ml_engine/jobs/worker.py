from __future__ import annotations

import datetime
import logging
import threading
import time
import uuid
from typing import Callable

from jobs.task_registry import get_task_handler

logger = logging.getLogger("jobs.worker")


class JobWorker:
    """
    Distributed Asynchronous Job Worker.
    Polls the database for pending jobs and executes registered task handlers.
    Can be run as a standalone process or as a background daemon thread.
    """

    def __init__(
        self,
        db_session_factory: Callable[[], Any],
        worker_id: str | None = None,
        poll_interval: float = 1.0,
    ):
        self.session_factory = db_session_factory
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def stop(self):
        """Signal worker to stop gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def start_in_background(self) -> threading.Thread:
        """Start worker loop in a background daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_loop, name=f"JobWorkerThread-{self.worker_id}", daemon=True
        )
        self._thread.start()
        logger.info(f"Started background job worker: {self.worker_id}")
        return self._thread

    def run_loop(self):
        """Continuous execution loop."""
        logger.info(f"Job worker {self.worker_id} entering polling loop.")
        while not self._stop_event.is_set():
            processed = self.run_once()
            if not processed:
                time.sleep(self.poll_interval)

    def run_once(self) -> bool:
        """
        Polls for a single pending job and executes it.
        Returns True if a job was found and executed, False otherwise.
        """
        from db_models.orm import Job

        db = self.session_factory()
        try:
            # Atomic job selection
            job = (
                db.query(Job)
                .filter(Job.status == "pending")
                .order_by(Job.priority.desc(), Job.created_at.asc())
                .first()
            )
            if not job:
                return False

            job_id = job.id
            task_name = job.task_name
            payload = job.payload_json or {}

            # Mark running
            job.status = "running"
            job.worker_id = self.worker_id
            job.started_at = datetime.datetime.now(datetime.timezone.utc)
            job.progress = 5.0
            db.commit()

            logger.info(f"Worker {self.worker_id} acquired job {job_id} ({task_name})")

            # Progress callback
            def update_progress(pct: float):
                sub_db = self.session_factory()
                try:
                    sub_job = sub_db.query(Job).filter(Job.id == job_id).first()
                    if sub_job and sub_job.status == "running":
                        sub_job.progress = min(max(pct, 0.0), 99.0)
                        sub_db.commit()
                finally:
                    sub_db.close()

            handler = get_task_handler(task_name)
            if not handler:
                job.status = "failed"
                job.error_message = f"No handler registered for task: {task_name}"
                job.completed_at = datetime.datetime.now(datetime.timezone.utc)
                db.commit()
                return True

            try:
                # Check for cancellation before executing
                db.refresh(job)
                if job.status == "cancelled":
                    logger.info(f"Job {job_id} was cancelled before execution.")
                    return True

                result = handler(payload, update_progress)

                # Re-check status in case it was cancelled during handler run
                db.refresh(job)
                if job.status == "cancelled":
                    logger.info(f"Job {job_id} was cancelled during execution.")
                    return True

                job.status = "completed"
                job.progress = 100.0
                job.result_json = result
                job.error_message = None
                job.completed_at = datetime.datetime.now(datetime.timezone.utc)
                db.commit()
                logger.info(f"Worker {self.worker_id} completed job {job_id}")

            except Exception as exc:
                logger.exception(f"Error executing job {job_id}: {exc}")
                db.refresh(job)
                job.status = "failed"
                job.error_message = str(exc)
                job.completed_at = datetime.datetime.now(datetime.timezone.utc)
                db.commit()

            return True

        finally:
            db.close()
