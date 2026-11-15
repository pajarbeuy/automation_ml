"""
API Contract Testing (Section 5 of TESTING_STRATEGY.md)

Verifikasi bahwa backend API FastAPI mematuhi spesifikasi kontrak:
- Status code HTTP yang benar (200, 201, 204, 400, 404, 422)
- Skema respon JSON yang valid
- Validasi tipe data dan field wajib (Pydantic models)
- Penanganan error aman pada resource ID yang tidak ditemukan
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import io
import pytest

# Pastikan apps/api dan packages/ml_engine ada di sys.path
api_path = Path(__file__).parent.parent / "apps" / "api"
sys.path.insert(0, str(api_path))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from database import Base, get_db


# Database in-memory khusus untuk pengujian kontrak API terisolasi
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def client():
    # Buat ulang tabel sebelum tiap tes
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    # Bersihkan tabel setelah tes selesai
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


# =============================================================================
# 1. SYSTEM HEALTH CONTRACT
# =============================================================================

def test_contract_health_check(client):
    """GET /health harus mengembalikan 200 OK dengan format status dan version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "version" in data


# =============================================================================
# 2. PROJECTS RESOURCE CONTRACTS
# =============================================================================

def test_contract_create_project_success(client):
    """POST /api/projects dengan payload valid harus mengembalikan 201 dan skema ProjectOut."""
    payload = {
        "name": "Customer Churn Prediction",
        "description": "Model untuk memprediksi churn pelanggan",
        "task_type": "classification",
    }
    response = client.post("/api/projects", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert "id" in data
    assert data["name"] == payload["name"]
    assert data["description"] == payload["description"]
    assert data["task_type"] == "classification"
    assert "created_at" in data


def test_contract_create_project_missing_required_fields(client):
    """POST /api/projects tanpa field 'name' yang wajib harus menghasilkan 422 Unprocessable Entity."""
    payload = {
        "description": "Tanpa field name",
    }
    response = client.post("/api/projects", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_contract_list_projects(client):
    """GET /api/projects harus mengembalikan 200 OK dan list array."""
    response = client.get("/api/projects")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_contract_get_project_not_found(client):
    """GET /api/projects/{non_existent_id} harus mengembalikan 404 Not Found."""
    response = client.get("/api/projects/non-existent-uuid-12345")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_contract_update_project(client):
    """PATCH /api/projects/{id} harus mengembalikan 200 OK dengan field yang diperbarui."""
    # Buat proyek dulu
    create_res = client.post("/api/projects", json={"name": "Old Project Name"})
    project_id = create_res.json()["id"]

    # Update nama
    patch_res = client.patch(f"/api/projects/{project_id}", json={"name": "New Project Name"})
    assert patch_res.status_code == 200
    assert patch_res.json()["name"] == "New Project Name"


def test_contract_delete_project(client):
    """DELETE /api/projects/{id} harus mengembalikan 204 No Content."""
    create_res = client.post("/api/projects", json={"name": "To Be Deleted"})
    project_id = create_res.json()["id"]

    del_res = client.delete(f"/api/projects/{project_id}")
    assert del_res.status_code == 204

    # Pastikan sudah 404 setelah dihapus
    get_res = client.get(f"/api/projects/{project_id}")
    assert get_res.status_code == 404


# =============================================================================
# 3. DATASETS RESOURCE CONTRACTS
# =============================================================================

def test_contract_upload_dataset_non_existent_project(client):
    """POST /api/projects/{id}/datasets dengan project_id tidak ada harus mengembalikan 404."""
    files = {"file": ("test.csv", "a,b,target\n1,2,0\n3,4,1", "text/csv")}
    data = {"target_column": "target"}
    response = client.post("/api/projects/unknown-proj/datasets", data=data, files=files)
    assert response.status_code == 404


def test_contract_upload_empty_dataset(client):
    """POST /api/projects/{id}/datasets dengan file kosong harus mengembalikan 400 Bad Request."""
    create_res = client.post("/api/projects", json={"name": "Empty Upload Test"})
    proj_id = create_res.json()["id"]

    files = {"file": ("empty.csv", "", "text/csv")}
    data = {"target_column": "target"}
    response = client.post(f"/api/projects/{proj_id}/datasets", data=data, files=files)
    assert response.status_code == 400
    assert "kosong" in response.json()["detail"].lower()


def test_contract_get_dataset_not_found(client):
    """GET /api/datasets/{non_existent_id} harus mengembalikan 404 Not Found."""
    response = client.get("/api/datasets/unknown-dataset-id")
    assert response.status_code == 404


def test_contract_get_dataset_profile_not_found(client):
    """GET /api/datasets/{non_existent_id}/profile harus mengembalikan 404 Not Found."""
    response = client.get("/api/datasets/unknown-dataset-id/profile")
    assert response.status_code == 404


def test_contract_get_dataset_recommendations_not_found(client):
    """GET /api/datasets/{non_existent_id}/recommendations harus mengembalikan 404 Not Found."""
    response = client.get("/api/datasets/unknown-dataset-id/recommendations")
    assert response.status_code == 404


# =============================================================================
# 4. DECISIONS RESOURCE CONTRACTS
# =============================================================================

def test_contract_submit_decision_missing_required_fields(client):
    """POST /api/recommendations/decision tanpa field yang dibutuhkan harus mengembalikan 422."""
    payload = {"invalid_key": "val"}
    response = client.post("/api/recommendations/decision", json=payload)
    assert response.status_code == 422


def test_contract_list_decisions_empty(client):
    """GET /api/projects/{project_id}/decisions harus mengembalikan list (bisa kosong 200 OK)."""
    response = client.get("/api/projects/unknown-proj/decisions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_contract_delete_non_existent_decision(client):
    """DELETE /api/decisions/{id} yang tidak ada harus mengembalikan 404 Not Found."""
    response = client.delete("/api/decisions/unknown-decision-id")
    assert response.status_code == 404


# =============================================================================
# 5. MODEL SERVING / PREDICT CONTRACTS (Section 5 of TESTING_STRATEGY)
# =============================================================================

def test_contract_predict_unknown_project(client):
    """POST /api/projects/{id}/predict pada project yang tidak ada harus mengembalikan 404."""
    payload = {"features": {"age": 30, "income": 50000}}
    response = client.post("/api/projects/unknown-project-id/predict", json=payload)
    assert response.status_code == 404
    assert "tidak ditemukan" in response.json()["detail"].lower() or "not found" in response.json()["detail"].lower()


def test_contract_predict_no_trained_model(client):
    """POST /api/projects/{id}/predict pada project tanpa model aktif harus mengembalikan 400 Bad Request."""
    create_res = client.post("/api/projects", json={"name": "No Model Project"})
    proj_id = create_res.json()["id"]

    payload = {"features": {"x1": 10}}
    response = client.post(f"/api/projects/{proj_id}/predict", json=payload)
    assert response.status_code == 400
    assert "kandidat" in response.json()["detail"].lower() or "model" in response.json()["detail"].lower()


def test_contract_predict_invalid_payload_type(client):
    """POST /api/projects/{id}/predict dengan payload bertipe invalid harus mengembalikan 422."""
    create_res = client.post("/api/projects", json={"name": "Payload Schema Test"})
    proj_id = create_res.json()["id"]

    # features harus bertipe object (dict), bukan string atau list
    response = client.post(f"/api/projects/{proj_id}/predict", json={"features": "invalid_string"})
    assert response.status_code == 422


# =============================================================================
# 6. ASSISTANT EXPLAINER CONTRACTS (WITH DATA LEAKAGE SUPPORT)
# =============================================================================

def test_contract_assistant_explain_recommendation_valid(client):
    """POST /api/assistant/explain-recommendation harus mengembalikan 200 dengan skema penjelasan lengkap."""
    payload = {
        "category": "data_leakage",
        "column_name": "target_proxy_col",
        "recommendation": "drop_column",
        "score": 95,
        "evidence": {"correlation_value": "0.998", "leakage_type": "target_proxy"},
        "rationale": "High correlation with label.",
        "alternative_actions": ["verify_manually"],
    }
    response = client.post("/api/assistant/explain-recommendation", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "headline" in data
    assert "detailed_explanation" in data
    assert "key_evidence" in data
    assert "alternative_rationale" in data
    assert data["category"] == "data_leakage"
    assert "Drop Column" in data["action_title"]


def test_contract_assistant_explain_missing_fields(client):
    """POST /api/assistant/explain-recommendation tanpa payload valid harus mengembalikan 422."""
    response = client.post("/api/assistant/explain-recommendation", json={"invalid": 123})
    assert response.status_code == 422


# =============================================================================
# 7. DRIFT MONITORING CONTRACTS
# =============================================================================

def test_contract_drift_summary_unknown_project(client):
    """GET /api/projects/{id}/drift/summary pada project yang tidak ada harus mengembalikan 404."""
    response = client.get("/api/projects/unknown-proj/drift/summary")
    assert response.status_code == 404


# =============================================================================
# 8. TREATMENT CANDIDATES & EMPIRICAL VALIDATION CONTRACTS
# =============================================================================

def test_contract_treatment_candidates_unknown_project(client):
    """GET /api/projects/{id}/treatment-candidates pada project yang tidak ada harus 404."""
    response = client.get("/api/projects/unknown-proj-id/treatment-candidates")
    assert response.status_code == 404


def test_contract_treatment_candidates_evaluate_unknown_project(client):
    """POST /api/projects/{id}/treatment-candidates/evaluate pada project yang tidak ada harus 404."""
    response = client.post("/api/projects/unknown-proj-id/treatment-candidates/evaluate")
    assert response.status_code == 404


def test_contract_treatment_candidates_apply_unknown_project(client):
    """POST /api/projects/{id}/treatment-candidates/apply pada project yang tidak ada harus 404."""
    response = client.post(
        "/api/projects/unknown-proj-id/treatment-candidates/apply",
        json={"candidate_id": "recommended"},
    )
    assert response.status_code == 404


def test_contract_treatment_candidates_full_lifecycle(client):
    """
    Siklus lengkap API Contract:
    Create Project -> Upload CSV -> Get Candidates -> Evaluate Candidates -> Apply Candidate
    """
    # 1. Buat Project
    proj_resp = client.post("/api/projects", json={"name": "Candidate Validation Test Project", "task_type": "classification"})
    assert proj_resp.status_code == 201
    proj_id = proj_resp.json()["id"]

    # 2. Upload CSV dataset (15 baris)
    csv_content = (
        "feature_a,feature_b,label\n"
        "1.2,cat1,0\n"
        "2.4,cat2,1\n"
        "1.8,cat1,0\n"
        "3.1,cat2,1\n"
        "2.0,cat1,0\n"
        "2.9,cat2,1\n"
        "1.5,cat1,0\n"
        "3.5,cat2,1\n"
        "2.2,cat1,0\n"
        "3.8,cat2,1\n"
        "1.1,cat1,0\n"
        "3.2,cat2,1\n"
        "1.9,cat1,0\n"
        "3.6,cat2,1\n"
        "2.5,cat2,1\n"
    )
    files = {"file": ("test_data.csv", csv_content, "text/csv")}
    upload_resp = client.post(
        f"/api/projects/{proj_id}/datasets",
        files=files,
        data={"target_column": "label"},
    )
    assert upload_resp.status_code == 201

    # 3. GET Treatment Candidates
    cand_resp = client.get(f"/api/projects/{proj_id}/treatment-candidates")
    assert cand_resp.status_code == 200
    candidates = cand_resp.json()
    assert isinstance(candidates, list)
    assert len(candidates) == 3
    cand_ids = [c["candidate_id"] for c in candidates]
    assert "recommended" in cand_ids
    assert "baseline" in cand_ids
    assert "alternative" in cand_ids

    # 4. POST Evaluate Treatment Candidates
    eval_resp = client.post(f"/api/projects/{proj_id}/treatment-candidates/evaluate")
    assert eval_resp.status_code == 200
    report = eval_resp.json()
    assert "validation_status" in report
    assert "lift" in report
    assert "benchmark_model" in report
    assert "ci_lower" in report
    assert "ci_upper" in report
    assert "mean_difference" in report
    assert "baseline_fold_scores" in report
    assert "recommended_fold_scores" in report
    assert "fold_differences" in report
    assert "evaluation_metadata" in report
    meta = report["evaluation_metadata"]
    assert meta["task_type"] == "classification"
    assert meta["ci_method"] == "paired_two_tailed_student_t"
    assert meta["degrees_of_freedom"] == 2
    assert meta["n_valid_folds"] == 3
    assert len(report["candidates"]) == 3

    # 5. POST Apply Candidate (Batch 1-click apply)
    apply_resp = client.post(
        f"/api/projects/{proj_id}/treatment-candidates/apply",
        json={"candidate_id": "recommended"},
    )
    assert apply_resp.status_code == 200
    apply_data = apply_resp.json()
    assert apply_data["status"] == "success"
    assert apply_data["candidate_id"] == "recommended"
    assert apply_data["applied_count"] > 0

    # 6. Verifikasi decisions tersimpan di database
    dec_resp = client.get(f"/api/projects/{proj_id}/decisions")
    assert dec_resp.status_code == 200
    saved_decisions = dec_resp.json()
    assert len(saved_decisions) == apply_data["applied_count"]


def test_contract_experiment_name(client):
    """Verifikasi bahwa experiment_name dapat dikirim dan dikembalikan oleh endpoint experiments."""
    create_res = client.post("/api/projects", json={"name": "Exp Name Test"})
    proj_id = create_res.json()["id"]

    csv_data = "feat1,feat2,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n"
    files = {"file": ("test.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    client.post(
        f"/api/projects/{proj_id}/datasets",
        files=files,
        data={"target_column": "label"},
    )

    exp_payload = {
        "model_names": ["logistic_regression"],
        "experiment_name": "Pipeline V1 - Baseline MinMax",
        "cv_folds": 2,
    }
    exp_resp = client.post(f"/api/projects/{proj_id}/experiments", json=exp_payload)
    assert exp_resp.status_code == 201
    exps = exp_resp.json()
    assert len(exps) == 1
    assert exps[0]["experiment_name"] == "Pipeline V1 - Baseline MinMax"

    # Verifikasi saat di-list
    list_resp = client.get(f"/api/projects/{proj_id}/experiments")
    assert list_resp.status_code == 200
    listed = list_resp.json()
    matching = [e for e in listed if e["id"] == exps[0]["id"]]
    assert len(matching) == 1
    assert matching[0]["experiment_name"] == "Pipeline V1 - Baseline MinMax"


