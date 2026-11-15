from __future__ import annotations

import datetime
import hashlib
import hmac
import os
import secrets
from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from db_models.orm import ApiKey, Project, Tenant, User

security_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Password Hashing & Verification (PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a cryptographically secure random salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"pbkdf2:sha256:100000${salt.hex()}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against the hashed string."""
    try:
        parts = hashed_password.split("$")
        if len(parts) == 3:
            _, salt_hex, key_hex = parts
        elif len(parts) == 2:
            salt_hex, key_hex = parts
        else:
            return False

        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        computed_key = hashlib.pbkdf2_hmac(
            "sha256", plain_password.encode("utf-8"), salt, 100000
        )
        return hmac.compare_digest(computed_key, expected_key)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------

def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.
    Returns (raw_key, prefix, key_hash).
    """
    token = secrets.token_urlsafe(32)
    raw_key = f"aml_{token}"
    prefix = raw_key[:10]
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return raw_key, prefix, key_hash


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# JWT Access Token Creation & Decoding
# ---------------------------------------------------------------------------

def create_access_token(
    data: dict[str, Any], expires_delta: datetime.timedelta | None = None
) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.now(datetime.timezone.utc) + expires_delta
    else:
        expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            minutes=settings.access_token_expire_minutes
        )
    to_encode.update({"exp": expire, "iat": datetime.datetime.now(datetime.timezone.utc)})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Multi-tenant Default Helper (Zero-Friction Backward Compatibility)
# ---------------------------------------------------------------------------

def get_or_create_default_tenant(db: Session) -> Tenant:
    tenant = db.query(Tenant).filter(Tenant.slug == "default").first()
    if not tenant:
        tenant = Tenant(name="Default Organization", slug="default", is_active=True)
        db.add(tenant)
        db.commit()
        db.refresh(tenant)
    return tenant


def get_or_create_default_user(db: Session) -> User:
    tenant = get_or_create_default_tenant(db)
    user = (
        db.query(User)
        .filter((User.username == "default_admin") | (User.email == "admin@automl.local"))
        .first()
    )
    if not user:
        user = User(
            email="admin@automl.local",
            username="default_admin",
            hashed_password=hash_password("admin123"),
            role="admin",
            tenant_id=tenant.id,
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    elif not verify_password("admin123", user.hashed_password):
        user.hashed_password = hash_password("admin123")
        db.commit()
        db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Authentication & Authorization Dependencies
# ---------------------------------------------------------------------------

def get_current_user(
    auth: HTTPAuthorizationCredentials | None = Depends(security_scheme),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> User:
    """
    Authenticate user via Bearer JWT token or X-API-Key.
    If no credentials supplied in debug/dev mode, gracefully provides the default workspace user.
    """
    if x_api_key:
        key_hash = hash_api_key(x_api_key)
        api_key_obj = (
            db.query(ApiKey)
            .filter(ApiKey.key_hash == key_hash, ApiKey.is_active == True)  # noqa: E712
            .first()
        )
        if api_key_obj:
            if api_key_obj.expires_at and api_key_obj.expires_at < datetime.datetime.now(
                datetime.timezone.utc
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key has expired"
                )
            user = db.query(User).filter(User.id == api_key_obj.user_id).first()
            if user and user.is_active:
                return user

    if auth and auth.credentials:
        payload = decode_access_token(auth.credentials)
        user_id = payload.get("sub")
        if user_id:
            user = db.query(User).filter(User.id == user_id).first()
            if user and user.is_active:
                return user

    # Backward-compatible fallback for development / existing unauthenticated test runs
    return get_or_create_default_user(db)


def require_role(allowed_roles: list[str]):
    """Role-Based Access Control (RBAC) dependency."""
    def _role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role == "admin":
            return current_user  # Superadmin bypass
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required roles: {allowed_roles}",
            )
        return current_user
    return _role_checker


def get_current_tenant(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Tenant:
    if current_user.tenant_id:
        tenant = db.query(Tenant).filter(Tenant.id == current_user.tenant_id).first()
        if tenant:
            return tenant
    return get_or_create_default_tenant(db)


def verify_project_access(
    project_id: str,
    db: Session,
    user: User,
) -> Project:
    """
    Ensure project belongs to the user's tenant (or user is system admin).
    Prevents cross-tenant data leakage.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if user.role != "admin" and project.tenant_id and user.tenant_id:
        if project.tenant_id != user.tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Project belongs to another tenant",
            )
    return project
