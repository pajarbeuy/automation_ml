from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from db_models import Dataset, WorkflowDecision, Project

from routers.utils import features_from_profile_json, dataset_profile_from_json

_PKG = str(Path(__file__).parent.parent.parent.parent / "packages" / "ml_engine")

if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from recommendations import (
    generate_recommendations,
    generate_treatment_candidates,
    evaluate_treatment_candidates,
)
from schemas import (
    RecommendationOut,
    DecisionIn,
    WorkflowDecisionOut,
    TreatmentCandidateOut,
    TreatmentValidationReportOut,
    ApplyCandidateIn,
    ApplyCandidateOut,
)

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

    project = db.query(Project).filter(Project.id == dataset.project_id).first()
    task_type = project.task_type if (project and project.task_type) else "classification"

    profile = dataset_profile_from_json(dataset.profile_json)
    df = pd.read_csv(dataset.storage_uri)

    recs = generate_recommendations(df, profile, task_type=task_type)
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


# =============================================================================
# CANDIDATE TREATMENT GENERATION, EVALUATION, & APPLY
# =============================================================================

def _get_project_and_dataset(project_id: str, db: Session) -> tuple[Project, Dataset]:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="No dataset found for this project")
    if not dataset.profile_json:
        raise HTTPException(status_code=400, detail="Dataset not yet profiled")

    return project, dataset


@router.get(
    "/projects/{project_id}/treatment-candidates",
    response_model=list[TreatmentCandidateOut],
)
def get_project_treatment_candidates(
    project_id: str, db: Session = Depends(get_db)
):
    """Ambil daftar varian strategi perlakuan data (Treatment Candidates)."""
    project, dataset = _get_project_and_dataset(project_id, db)
    task_type = project.task_type or "classification"
    profile = dataset_profile_from_json(dataset.profile_json)
    df = pd.read_csv(dataset.storage_uri)

    recs = generate_recommendations(df, profile, task_type=task_type)
    candidates = generate_treatment_candidates(profile, recs, task_type=task_type)
    return [c.to_dict() for c in candidates]


@router.post(
    "/projects/{project_id}/treatment-candidates/evaluate",
    response_model=TreatmentValidationReportOut,
)
def evaluate_project_treatment_candidates(
    project_id: str,
    confidence_level: float = 0.90,
    db: Session = Depends(get_db),
):
    """
    Evaluasi seluruh Treatment Candidates secara objektif dengan K-Fold CV
    menggunakan Fast Reference Estimator dan Paired Difference CI untuk menghitung Treatment Lift empiris.
    """
    project, dataset = _get_project_and_dataset(project_id, db)
    if not project.target_column:
        raise HTTPException(status_code=400, detail="Project target column is not configured")

    task_type = project.task_type or "classification"
    profile = dataset_profile_from_json(dataset.profile_json)
    df = pd.read_csv(dataset.storage_uri)

    recs = generate_recommendations(df, profile, task_type=task_type)
    candidates = generate_treatment_candidates(profile, recs, task_type=task_type)

    report = evaluate_treatment_candidates(
        df=df,
        profile=profile,
        candidates=candidates,
        target_col=project.target_column,
        task_type=task_type,
        n_splits=3,
        seed=42,
        confidence_level=confidence_level,
    )
    return report.to_dict()


@router.post(
    "/projects/{project_id}/treatment-candidates/apply",
    response_model=ApplyCandidateOut,
)
def apply_project_treatment_candidate(
    project_id: str, payload: ApplyCandidateIn, db: Session = Depends(get_db)
):
    """
    Terapkan seluruh keputusan dari kandidat strategi terpilih secara atomik
    (One-Click Batch Apply).
    """
    project, dataset = _get_project_and_dataset(project_id, db)
    task_type = project.task_type or "classification"
    profile = dataset_profile_from_json(dataset.profile_json)
    df = pd.read_csv(dataset.storage_uri)

    recs = generate_recommendations(df, profile, task_type=task_type)
    candidates = generate_treatment_candidates(profile, recs, task_type=task_type)

    target_cand = next((c for c in candidates if c.candidate_id == payload.candidate_id), None)
    if not target_cand:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate treatment '{payload.candidate_id}' not found",
        )

    rec_id_map = {(r.category, r.column): r.id for r in recs}
    applied_decisions: list[WorkflowDecision] = []
    for d_item in target_cand.decisions:
        cat = d_item.get("category")
        col = d_item.get("column_name")
        action = d_item.get("selected_action")
        rec_id = rec_id_map.get((cat, col))

        existing = (
            db.query(WorkflowDecision)
            .filter(
                WorkflowDecision.project_id == project_id,
                WorkflowDecision.category == cat,
                WorkflowDecision.column_name == col,
            )
            .first()
        )
        if existing:
            existing.previous_action = existing.selected_action
            existing.selected_action = action
            existing.status = "accepted"
            existing.rationale = f"Diterapkan dari {target_cand.name}"
            if not existing.recommendation_id and rec_id:
                existing.recommendation_id = rec_id
            applied_decisions.append(existing)
        else:
            new_d = WorkflowDecision(
                id=str(uuid.uuid4()),
                project_id=project_id,
                recommendation_id=rec_id,
                category=cat,
                column_name=col,
                selected_action=action,
                status="accepted",
                rationale=f"Diterapkan dari {target_cand.name}",
            )
            db.add(new_d)
            applied_decisions.append(new_d)

    db.commit()
    for d in applied_decisions:
        db.refresh(d)

    return {
        "status": "success",
        "candidate_id": payload.candidate_id,
        "applied_count": len(applied_decisions),
        "decisions": applied_decisions,
    }

