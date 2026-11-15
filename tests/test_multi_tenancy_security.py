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
from db_models.orm import Project, Tenant, User
from main import app
from security import (
    create_access_token,
    decode_access_token,
    generate_api_key,
    hash_password,
    verify_password,
)

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


def test_password_hashing():
    pw = "SuperSecretPassword123!"
    hashed = hash_password(pw)
    assert hashed != pw
    assert verify_password(pw, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_jwt_token_roundtrip():
    data = {"sub": "user-uuid-123", "role": "admin", "tenant_id": "tenant-456"}
    token = create_access_token(data)
    decoded = decode_access_token(token)
    assert decoded["sub"] == "user-uuid-123"
    assert decoded["role"] == "admin"
    assert decoded["tenant_id"] == "tenant-456"


def test_api_key_generation():
    raw_key, prefix, key_hash = generate_api_key()
    assert raw_key.startswith("aml_")
    assert len(raw_key) > 30
    assert prefix == raw_key[:10]
    assert len(key_hash) == 64


def test_auth_registration_and_login_flow(client):
    # 1. Register
    reg_resp = client.post(
        "/api/auth/register",
        json={
            "email": "user1@enterprise.com",
            "username": "user1",
            "password": "Password123",
            "role": "data_scientist",
        },
    )
    assert reg_resp.status_code == 201
    user_data = reg_resp.json()
    assert user_data["email"] == "user1@enterprise.com"

    # 2. Login
    login_resp = client.post(
        "/api/auth/login",
        json={"email": "user1@enterprise.com", "password": "Password123"},
    )
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # 3. Access profile with Bearer Token
    me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "user1"


def test_multi_tenant_project_isolation(client):
    db = TestingSessionLocal()
    # Create 2 tenants
    t1 = Tenant(name="Acme Corp", slug="acme")
    t2 = Tenant(name="Beta Inc", slug="beta")
    db.add_all([t1, t2])
    db.commit()

    # Create 2 users in different tenants
    u1 = User(
        email="alice@acme.com",
        username="alice",
        hashed_password=hash_password("pw1"),
        tenant_id=t1.id,
        role="data_scientist",
    )
    u2 = User(
        email="bob@beta.com",
        username="bob",
        hashed_password=hash_password("pw2"),
        tenant_id=t2.id,
        role="data_scientist",
    )
    db.add_all([u1, u2])
    db.commit()

    # User 1 creates project
    token1 = create_access_token({"sub": u1.id, "tenant_id": t1.id, "role": "data_scientist"})
    p1_resp = client.post(
        "/api/projects",
        json={"name": "Acme Churn Model", "task_type": "classification"},
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert p1_resp.status_code == 201
    project1_id = p1_resp.json()["id"]

    # User 2 logs in
    token2 = create_access_token({"sub": u2.id, "tenant_id": t2.id, "role": "data_scientist"})

    # User 2 tries to access User 1's project -> MUST BE FORBIDDEN (403)
    p_get_resp = client.get(
        f"/api/projects/{project1_id}",
        headers={"Authorization": f"Bearer {token2}"},
    )
    assert p_get_resp.status_code == 403
    assert "belongs to another tenant" in p_get_resp.json()["detail"]

    # User 1 can access their own project
    p1_get = client.get(
        f"/api/projects/{project1_id}",
        headers={"Authorization": f"Bearer {token1}"},
    )
    assert p1_get.status_code == 200
    db.close()
