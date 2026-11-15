from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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
    experiment_name: str | None = None
    random_seed: int = 42
    cv_folds: int = 5
    primary_metric: str = "f1"
    test_size: float = 0.2
    enable_feature_engineering: bool = False
    is_tuned: bool = False
    model_params: dict[str, Any] | None = None


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
    experiment_name: str | None = None
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


# ---------------------------------------------------------------------------
# Treatment Candidates & Validation
# ---------------------------------------------------------------------------

class TreatmentCandidateOut(BaseModel):
    candidate_id: str
    name: str
    strategy_type: str
    description: str
    decisions: list[dict[str, Any]]
    summary_tags: list[str]


class TreatmentCandidateEvaluationOut(BaseModel):
    candidate_id: str
    name: str
    strategy_type: str
    primary_metric_name: str
    primary_score: float
    primary_std: float
    secondary_metric_name: str
    secondary_score: float
    rank: int
    summary_tags: list[str]
    training_time_seconds: float


class TreatmentValidationReportOut(BaseModel):
    task_type: str
    benchmark_model: str
    primary_metric: str
    validation_status: Literal["SUPERIOR", "INCONCLUSIVE", "SUBOPTIMAL", "INVALID_EVALUATION"]
    headline: str
    explanation: str
    lift: float | None = None
    lift_percentage: float | None = None
    mean_difference: float | None = None
    relative_difference: float | None = None
    best_candidate_id: str
    confidence_level: float = 0.90
    ci_lower: float | None = None
    ci_upper: float | None = None
    baseline_fold_scores: list[float] = []
    recommended_fold_scores: list[float] = []
    fold_differences: list[float] = []
    sign_agreement: str = ""
    evaluation_metadata: dict[str, Any] = {}
    candidates: list[TreatmentCandidateEvaluationOut]


class ApplyCandidateIn(BaseModel):
    candidate_id: str


class ApplyCandidateOut(BaseModel):
    status: str = "success"
    candidate_id: str
    applied_count: int
    decisions: list[WorkflowDecisionOut]



# ---------------------------------------------------------------------------
# Multi-Tenant & Security Schemas
# ---------------------------------------------------------------------------

class TenantCreate(BaseModel):
    name: str
    slug: str


class TenantOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    slug: str
    is_active: bool
    created_at: datetime


class UserRegister(BaseModel):
    email: str
    username: str
    password: str
    tenant_id: str | None = None
    role: str = "data_scientist"


class UserLogin(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    username: str
    role: str
    tenant_id: str | None = None
    is_active: bool
    created_at: datetime


class ApiKeyCreate(BaseModel):
    name: str
    expires_in_days: int | None = None


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    prefix: str
    created_at: datetime
    expires_at: datetime | None = None
    raw_key: str | None = None  # Returned once on creation


# ---------------------------------------------------------------------------
# Asynchronous Job Worker Schemas
# ---------------------------------------------------------------------------

class JobEnqueueIn(BaseModel):
    task_name: str
    project_id: str | None = None
    payload: dict[str, Any] = {}
    priority: int = 0


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str | None = None
    tenant_id: str | None = None
    task_name: str
    status: str
    priority: int
    progress: float
    payload_json: dict[str, Any] | None = None
    result_json: dict[str, Any] | None = None
    error_message: str | None = None
    worker_id: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Cloud Object Storage Schemas
# ---------------------------------------------------------------------------

class StorageConfigOut(BaseModel):
    backend: str
    is_cloud: bool
    base_uri: str


# ---------------------------------------------------------------------------
# Meta-Learning Recommendation Ranker Schemas
# ---------------------------------------------------------------------------

class MetaRankCandidateItem(BaseModel):
    candidate_id: str
    name: str
    strategy_type: str
    predicted_lift: float
    predicted_score: float
    meta_confidence: float
    heuristic_score: float
    blended_rank_score: float
    rationale: str


class MetaRankReportOut(BaseModel):
    project_id: str
    task_type: str
    meta_features: dict[str, Any]
    model_status: str  # "trained" | "cold_start_fallback"
    historical_runs_count: int
    ranked_candidates: list[MetaRankCandidateItem]


class MetaTrainIn(BaseModel):
    force_retrain: bool = False


class MetaTrainOut(BaseModel):
    status: str
    total_samples: int
    features_used: list[str]
    test_r2_score: float | None = None
    message: str
