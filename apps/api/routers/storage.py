from __future__ import annotations

from fastapi import APIRouter, Depends
from config import settings
from db_models.orm import User
from schemas.schemas import StorageConfigOut
from security import get_current_user
from storage.factory import get_storage_adapter

router = APIRouter(prefix="/storage", tags=["Cloud Object Storage"])


@router.get("/config", response_model=StorageConfigOut)
def get_storage_configuration(
    current_user: User = Depends(get_current_user),
):
    """Retrieve active storage backend configuration and cloud status."""
    adapter = get_storage_adapter(
        backend=settings.storage_backend,
        base_path=settings.storage_path,
        bucket=settings.s3_bucket,
        endpoint_url=settings.s3_endpoint_url,
        region=settings.s3_region,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        prefix=settings.s3_prefix,
    )
    is_cloud = settings.storage_backend in ("s3", "minio", "cloud")
    base_uri = f"s3://{settings.s3_bucket}" if is_cloud else f"file:///{settings.storage_path}"

    return {
        "backend": settings.storage_backend,
        "is_cloud": is_cloud,
        "base_uri": base_uri,
    }


@router.get("/url")
def get_presigned_url(
    uri: str,
    expires_in: int = 3600,
    current_user: User = Depends(get_current_user),
):
    """Generate a direct or presigned URL for downloading a dataset or model artifact."""
    adapter = get_storage_adapter(
        backend=settings.storage_backend,
        base_path=settings.storage_path,
        bucket=settings.s3_bucket,
        endpoint_url=settings.s3_endpoint_url,
        region=settings.s3_region,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        prefix=settings.s3_prefix,
    )
    url = adapter.get_url(uri, expires_in=expires_in)
    return {"uri": uri, "url": url, "expires_in": expires_in}
