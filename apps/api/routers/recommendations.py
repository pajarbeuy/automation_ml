from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from db_models import Dataset, WorkflowDecision

from routers.utils import features_from_profile_json, dataset_profile_from_json

_PKG = str(Path(__file__).parent.parent.parent.parent / "packages" / "ml_engine")

if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from recommendations.engine import generate_recommendations
from schemas import RecommendationOut, DecisionIn, WorkflowDecisionOut

router = APIRouter(tags=["recommendations"])





@router.get(
    "/datasets/{dataset_id}/recommendations",
    response_model=list[RecommendationOut],
)
def get_recommendations(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset.profile_json:
        raise HTTPException(status_code=400, detail="Dataset not yet profiled")

    profile = dataset_profile_from_json(dataset.profile_json)
    df = pd.read_csv(dataset.storage_uri)

    recs = generate_recommendations(df, profile)
    return [rec.to_dict() for rec in recs]


@router.post(
    "/recommendations/decision",
    response_model=WorkflowDecisionOut,
    status_code=status.HTTP_201_CREATED,
)
def submit_decision(payload: DecisionIn, db: Session = Depends(get_db)):
    """Accept, reject, or override a recommendation."""
    decision = WorkflowDecision(
        id=str(uuid.uuid4()),
        project_id=_resolve_project_from_recommendation(payload.recommendation_id, db),
        recommendation_id=payload.recommendation_id,
        category=payload.category,
        column_name=payload.column_name,
        selected_action=payload.selected_action,
        status=payload.status,
        rationale=payload.rationale,
    )
    db.add(decision)
    db.commit()
    db.refresh(decision)
    return decision


@router.get(
    "/projects/{project_id}/decisions",
    response_model=list[WorkflowDecisionOut],
)
def list_decisions(project_id: str, db: Session = Depends(get_db)):
    return (
        db.query(WorkflowDecision)
        .filter(WorkflowDecision.project_id == project_id)
        .all()
    )


@router.delete("/decisions/{decision_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_decision(decision_id: str, db: Session = Depends(get_db)):
    decision = db.query(WorkflowDecision).filter(WorkflowDecision.id == decision_id).first()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    db.delete(decision)
    db.commit()


def _resolve_project_from_recommendation(rec_id: str | None, db: Session) -> str:
    """
    Placeholder: in a real system the recommendation_id would link to a stored rec.
    For MVP, we pass project_id directly in the decision payload (via DecisionIn extension).
    """
    return "unknown"  # Will be overridden by the frontend passing project_id


@router.post(
    "/projects/{project_id}/decisions",
    response_model=WorkflowDecisionOut,
    status_code=status.HTTP_201_CREATED,
)
def submit_project_decision(
    project_id: str, payload: DecisionIn, db: Session = Depends(get_db)
):
    """
    Submit a decision scoped to a project.
    This is the primary endpoint the frontend uses.
    """
    # Upsert: replace existing decision for the same column + category
    existing = (
        db.query(WorkflowDecision)
        .filter(
            WorkflowDecision.project_id == project_id,
            WorkflowDecision.category == payload.category,
            WorkflowDecision.column_name == payload.column_name,
        )
        .first()
    )

    if existing:
        existing.previous_action = existing.selected_action
        existing.selected_action = payload.selected_action
        existing.status = payload.status
        existing.rationale = payload.rationale
        existing.recommendation_id = payload.recommendation_id
        db.commit()
        db.refresh(existing)
        return existing

    decision = WorkflowDecision(
        project_id=project_id,
        recommendation_id=payload.recommendation_id,
        category=payload.category,
        column_name=payload.column_name,
        selected_action=payload.selected_action,
        status=payload.status,
        rationale=payload.rationale,
    )
    db.add(decision)
    db.commit()
    db.refresh(decision)
    return decision
