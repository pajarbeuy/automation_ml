from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    task_type: Mapped[str] = mapped_column(String(50), default="classification")
    target_column: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset", back_populates="project", cascade="all, delete-orphan"
    )
    decisions: Mapped[list["WorkflowDecision"]] = relationship(
        "WorkflowDecision", back_populates="project", cascade="all, delete-orphan"
    )
    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment", back_populates="project", cascade="all, delete-orphan"
    )
    pipeline_config: Mapped["PipelineConfig | None"] = relationship(
        "PipelineConfig", back_populates="project", cascade="all, delete-orphan", uselist=False
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(512))
    storage_uri: Mapped[str] = mapped_column(String(1024))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    rows: Mapped[int | None] = mapped_column(Integer)
    columns: Mapped[int | None] = mapped_column(Integer)
    schema_json: Mapped[dict | None] = mapped_column(JSON)
    profile_json: Mapped[dict | None] = mapped_column(JSON)
    eda_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped["Project"] = relationship("Project", back_populates="datasets")
    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment", back_populates="dataset"
    )


class WorkflowDecision(Base):
    __tablename__ = "workflow_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    recommendation_id: Mapped[str | None] = mapped_column(String(36))
    category: Mapped[str] = mapped_column(String(100))  # missing_value, outlier, encoding, scaling
    column_name: Mapped[str | None] = mapped_column(String(255))
    selected_action: Mapped[str] = mapped_column(String(100))
    previous_action: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(
        String(50), default="proposed"
    )  # proposed/accepted/rejected/overridden
    rationale: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped["Project"] = relationship("Project", back_populates="decisions")


class PipelineConfig(Base):
    __tablename__ = "pipeline_configs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), unique=True, nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    pipeline_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    project: Mapped["Project"] = relationship("Project", back_populates="pipeline_config")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    pipeline_hash: Mapped[str | None] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(100))
    random_seed: Mapped[int] = mapped_column(Integer, default=42)
    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending/running/completed/failed
    error_message: Mapped[str | None] = mapped_column(Text)
    training_time_seconds: Mapped[float | None] = mapped_column(Float)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped["Project"] = relationship("Project", back_populates="experiments")
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="experiments")
    metrics: Mapped[list["Metric"]] = relationship(
        "Metric", back_populates="experiment", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list["Artifact"]] = relationship(
        "Artifact", back_populates="experiment", cascade="all, delete-orphan"
    )


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100))
    value: Mapped[float] = mapped_column(Float)
    split: Mapped[str | None] = mapped_column(String(50))  # train/test/cv
    fold: Mapped[int | None] = mapped_column(Integer)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="metrics")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), nullable=False)
    type: Mapped[str] = mapped_column(String(100))  # model/pipeline/config/report
    storage_uri: Mapped[str] = mapped_column(String(1024))
    checksum: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="artifacts")
