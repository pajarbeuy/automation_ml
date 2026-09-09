from __future__ import annotations

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://automl:automl_secret@localhost:5432/automl_db"
    storage_path: str = "./storage"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    debug: bool = True
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
