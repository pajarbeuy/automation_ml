from __future__ import annotations

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://automl:automl_secret@localhost:5432/automl_db"
    storage_path: str = "./storage"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    debug: bool = True

    # Security & Multi-tenancy
    jwt_secret_key: str = "automl-super-secret-key-change-in-production-2026"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # Cloud Object Storage (S3 / MinIO / Local)
    storage_backend: str = "local"  # "local" or "s3"
    s3_endpoint_url: str = ""       # e.g. "http://localhost:9000" for MinIO
    s3_bucket: str = "automl-storage"
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_prefix: str = ""

    # Background Job Worker
    worker_concurrency: int = 2
    worker_poll_interval_seconds: float = 1.0

    # MLflow — set to a tracking URI to enable (e.g. "http://localhost:5000")
    # Leave empty/unset to disable MLflow logging
    mlflow_tracking_uri: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
