from __future__ import annotations

import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from db_models.orm import ApiKey, Tenant, User
from schemas.schemas import (
    ApiKeyCreate,
    ApiKeyOut,
    TenantCreate,
    TenantOut,
    TokenOut,
    UserLogin,
    UserOut,
    UserRegister,
)
from security import (
    create_access_token,
    generate_api_key,
    get_current_tenant,
    get_current_user,
    get_or_create_default_tenant,
    get_or_create_default_user,
    hash_password,
    require_role,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["Authentication & Security"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register_user(
    body: UserRegister,
    db: Session = Depends(get_db),
):
    """Register a new user inside a tenant workspace."""
    existing = db.query(User).filter((User.email == body.email) | (User.username == body.username)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists",
        )

    tenant_id = body.tenant_id
    if not tenant_id:
        default_tenant = get_or_create_default_tenant(db)
        tenant_id = default_tenant.id
    else:
        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant workspace not found")

    user = User(
        email=body.email,
        username=body.username,
        hashed_password=hash_password(body.password),
        tenant_id=tenant_id,
        role=body.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenOut)
def login_user(
    body: UserLogin,
    db: Session = Depends(get_db),
):
    """Authenticate with email and password to receive a JWT access token."""
    clean_email = body.email.strip()
    user = db.query(User).filter(User.email == clean_email).first()
    if not user and clean_email.lower() in ("admin@automl.local", "default_admin"):
        user = get_or_create_default_user(db)

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated")

    access_token = create_access_token(
        data={"sub": user.id, "email": user.email, "role": user.role, "tenant_id": user.tenant_id}
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
    }


@router.get("/me", response_model=UserOut)
def get_profile(
    current_user: User = Depends(get_current_user),
):
    """Get profile of current authenticated user."""
    return current_user


@router.post("/api-keys", response_model=ApiKeyOut, status_code=status.HTTP_201_CREATED)
def create_user_api_key(
    body: ApiKeyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a new programmatic API key for automated workflows."""
    raw_key, prefix, key_hash = generate_api_key()
    expires_at = None
    if body.expires_in_days:
        expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            days=body.expires_in_days
        )

    api_key_obj = ApiKey(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        name=body.name,
        prefix=prefix,
        key_hash=key_hash,
        expires_at=expires_at,
        is_active=True,
    )
    db.add(api_key_obj)
    db.commit()
    db.refresh(api_key_obj)

    out = ApiKeyOut.model_validate(api_key_obj)
    out.raw_key = raw_key
    return out


@router.get("/tenants", response_model=list[TenantOut])
def list_tenants(
    current_user: User = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    """List all tenant organizations (Admin only)."""
    return db.query(Tenant).all()


@router.post("/tenants", response_model=TenantOut, status_code=status.HTTP_201_CREATED)
def create_tenant(
    body: TenantCreate,
    current_user: User = Depends(require_role(["admin"])),
    db: Session = Depends(get_db),
):
    """Create a new tenant organization (Admin only)."""
    existing = db.query(Tenant).filter(Tenant.slug == body.slug).first()
    if existing:
        raise HTTPException(status_code=400, detail="Tenant with this slug already exists")

    tenant = Tenant(name=body.name, slug=body.slug, is_active=True)
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant
