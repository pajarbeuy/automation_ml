from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    task_type: str = "classification"


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    target_column: str | None = None
    status: str | None = None


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None
    task_type: str
    target_column: str | None
    status: str
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    filename: str
    content_hash: str | None
    rows: int | None
    columns: int | None
    created_at: datetime


class DatasetProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    rows: int | None
    columns: int | None
    profile_json: dict | None


class DatasetEDAOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    eda_json: dict | None


# ---------------------------------------------------------------------------
# Recommendation / Decision
# ---------------------------------------------------------------------------

class RecommendationOut(BaseModel):
    id: str
    category: str
    column: str | None
    recommended_action: str
    confidence: float
    score: float
    evidence: list[str]
    rationale: str
    alternatives: list[str]
    status: str


class DecisionIn(BaseModel):
    recommendation_id: str
    category: str
    column_name: str | None = None
    selected_action: str
    rationale: str | None = None
    status: str = "accepted"


class WorkflowDecisionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    recommendation_id: str | None
    category: str
    column_name: str | None
    selected_action: str
    status: str
    rationale: str | None
    created_at: datetime


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PipelineConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    config_json: dict
    pipeline_hash: str | None
    updated_at: datetime


class PipelineValidationOut(BaseModel):
    valid: bool
    steps: list[str]
    warnings: list[str]
    errors: list[str]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class ExperimentCreate(BaseModel):
    model_names: list[str]
    random_seed: int = 42
    cv_folds: int = 5
    primary_metric: str = "f1"
    test_size: float = 0.2
    enable_feature_engineering: bool = False
    is_tuned: bool = False


class MetricOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    value: float
    split: str | None
    fold: int | None


class ArtifactOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    type: str
    storage_uri: str
    checksum: str | None
    created_at: datetime


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    dataset_id: str
    model_name: str
    random_seed: int
    status: str
    training_time_seconds: float | None
    error_message: str | None
    config_json: dict
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    metrics: list[MetricOut] = []
    artifacts: list[ArtifactOut] = []


class HyperparameterTuneCreate(BaseModel):
    model_name: str
    n_trials: int = 10
    random_seed: int = 42
    cv_folds: int = 5
    primary_metric: str = "f1_weighted"


class HyperparameterTuneResultOut(BaseModel):
    model_name: str
    best_params: dict[str, Any]
    best_score: float
    primary_metric: str
    trials: list[dict[str, Any]]
    total_time_seconds: float


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------

class ErrorOut(BaseModel):
    status: str = "error"
    stage: str | None = None
    error_code: str | None = None
    message: str


class PaginatedOut(BaseModel):
    total: int
    items: list[Any]

