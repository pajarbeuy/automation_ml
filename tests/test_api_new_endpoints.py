from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

api_path = Path(__file__).parent.parent / "apps" / "api"
sys.path.insert(0, str(api_path))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))

from database import Base, get_db
from main import app

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


@pytest.fixture
def client():
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_storage_endpoints(client):
    res = client.get("/api/storage/config")
    assert res.status_code == 200
    data = res.json()
    assert "backend" in data
    assert "is_cloud" in data

    url_res = client.get("/api/storage/url?uri=datasets/test.csv")
    assert url_res.status_code == 200
    url_data = url_res.json()
    assert "url" in url_data
    assert url_data["uri"] == "datasets/test.csv"


def test_jobs_endpoints_and_lifecycle(client):
    # Enqueue job
    enq_res = client.post(
        "/api/jobs/enqueue",
        json={"task_name": "echo", "payload": {"message": "hello async"}, "priority": 10},
    )
    assert enq_res.status_code == 202
    job_data = enq_res.json()
    job_id = job_data["id"]
    assert job_data["status"] == "pending"
    assert job_data["task_name"] == "echo"

    # Poll status
    get_res = client.get(f"/api/jobs/{job_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == job_id

    # Cancel job
    cancel_res = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel_res.status_code == 200
    assert cancel_res.json()["status"] == "cancelled"


def test_meta_learning_endpoints(client):
    # Check status
    st_res = client.get("/api/meta-learning/status")
    assert st_res.status_code == 200
    st_data = st_res.json()
    assert "is_trained" in st_data
    assert "total_historical_runs" in st_data

    # Train (with 0 records -> falls back to cold-start fallback)
    train_res = client.post("/api/meta-learning/train", json={"force_retrain": True})
    assert train_res.status_code == 200
    train_data = train_res.json()
    assert train_data["status"] == "cold_start_fallback"
    assert "Cold-start heuristic retained" in train_data["message"]
