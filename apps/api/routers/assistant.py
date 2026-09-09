from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Add packages/ml_engine to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "packages" / "ml_engine"))

from database import get_db
from db_models.orm import Dataset, Experiment, Project, WorkflowDecision
from assistant import explain_recommendation, summarize_experiments, suggest_next_step

router = APIRouter(tags=["assistant"])


class RecommendationExplainRequest(BaseModel):
    category: str
    column_name: str | None = None
    selected_action: str | None = None
    recommendation: str | None = None
    score: float | int | None = 0
    evidence: dict[str, Any] | None = None
    rationale: str | None = None
    alternative_actions: list[str] | None = None


@router.post("/assistant/explain-recommendation")
def get_recommendation_explanation(payload: RecommendationExplainRequest):
    """Get natural language explanation for a preprocessing recommendation."""
    rec_dict = {
        "category": payload.category,
        "column_name": payload.column_name or "dataset",
        "recommendation": payload.recommendation or payload.selected_action or "",
        "score": payload.score or 0,
        "evidence": payload.evidence or {},
        "rationale": payload.rationale or "",
        "alternative_actions": payload.alternative_actions or [],
    }
    return explain_recommendation(rec_dict)


@router.get("/projects/{project_id}/assistant/summary")
def get_project_experiment_summary(project_id: str, db: Session = Depends(get_db)):
    """Get structured natural language summary for a project's experiments."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    experiments = (
        db.query(Experiment)
        .filter(Experiment.project_id == project_id)
        .order_by(Experiment.created_at.desc())
        .all()
    )

    candidate_exp = None
    if project.status and project.status.startswith("candidate:"):
        candidate_id = project.status.replace("candidate:", "")
        candidate_exp = db.query(Experiment).filter(Experiment.id == candidate_id).first()

    def _exp_dict(e: Experiment) -> dict:
        from db_models.orm import Metric
        metrics = db.query(Metric).filter(Metric.experiment_id == e.id, Metric.split == "cv").all()
        return {
            "model_name": e.model_name,
            "status": e.status,
            "training_time_seconds": e.training_time_seconds,
            "metrics": [{"name": m.name, "value": m.value, "split": m.split} for m in metrics],
        }

    exps_dicts = [_exp_dict(e) for e in experiments]
    candidate_dict = _exp_dict(candidate_exp) if candidate_exp else None

    return summarize_experiments(
        experiments=exps_dicts,
        candidate=candidate_dict,
        target_column=project.target_column or "target",
        task_type=project.task_type or "classification",
    )


@router.get("/projects/{project_id}/assistant/next-step")
def get_next_step_guidance(
    project_id: str,
    current_page: str = "upload",
    db: Session = Depends(get_db),
):
    """Get contextual next-step guidance for the user."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .first()
    )
    decisions = (
        db.query(WorkflowDecision)
        .filter(WorkflowDecision.project_id == project_id)
        .all()
    )
    experiments = (
        db.query(Experiment)
        .filter(Experiment.project_id == project_id)
        .all()
    )

    candidate_selected = bool(project.status and project.status.startswith("candidate:"))

    return suggest_next_step(
        current_page=current_page,
        has_dataset=dataset is not None,
        has_decisions=len(decisions) > 0,
        has_pipeline=len(decisions) > 0,
        has_experiments=len(experiments) > 0,
        candidate_selected=candidate_selected,
    )
