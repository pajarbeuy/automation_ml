from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from db_models.orm import Dataset, MetaRunRecord, Project, User
from recommendations.candidate_evaluator import generate_treatment_candidates
from recommendations.meta_ranker import MetaFeatureExtractor, get_meta_ranker
from schemas.schemas import MetaRankReportOut, MetaTrainIn, MetaTrainOut
from security import get_current_user, require_role, verify_project_access

router = APIRouter(prefix="/meta-learning", tags=["Meta-Learning Recommendation Ranker"])


@router.get("/projects/{project_id}/rank", response_model=MetaRankReportOut)
def get_meta_learning_ranking(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Extract meta-features from dataset profile and rank candidate treatment strategies
    using the meta-learning surrogate model blended with empirical heuristic scores.
    """
    project = verify_project_access(project_id, db, current_user)
    dataset = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .first()
    )
    if not dataset or not dataset.profile_json:
        raise HTTPException(
            status_code=400,
            detail="Project has no profiled dataset. Upload and profile a dataset first.",
        )

    meta_features = MetaFeatureExtractor.extract(dataset.profile_json, project.task_type)

    # Generate candidate strategies
    candidates = generate_treatment_candidates(
        profile=dataset.profile_json,
        task_type=project.task_type,
        target_col=project.target_column,
    )

    ranker = get_meta_ranker()
    ranked_candidates = ranker.rank_candidates(
        meta_features=meta_features,
        candidates=[
            {
                "candidate_id": c.candidate_id,
                "name": c.name,
                "strategy_type": c.strategy_type,
                "heuristic_score": 90.0 if c.strategy_type == "recommended" else (80.0 if c.strategy_type == "alternative" else 70.0),
            }
            for c in candidates
        ],
    )

    history_count = db.query(MetaRunRecord).count()

    return {
        "project_id": project_id,
        "task_type": project.task_type,
        "meta_features": meta_features,
        "model_status": "trained" if ranker.is_trained else "cold_start_fallback",
        "historical_runs_count": history_count,
        "ranked_candidates": ranked_candidates,
    }


@router.post("/train", response_model=MetaTrainOut)
def train_meta_learning_model(
    payload: MetaTrainIn = MetaTrainIn(),
    current_user: User = Depends(require_role(["admin", "data_scientist"])),
    db: Session = Depends(get_db),
):
    """
    Train or update the meta-learning surrogate ranker model using all historical runs
    recorded across the platform.
    """
    records = db.query(MetaRunRecord).all()
    record_dicts = [
        {
            "meta_features": r.meta_features_json,
            "strategy_name": r.strategy_name,
            "lift": r.lift,
        }
        for r in records
    ]

    ranker = get_meta_ranker()
    result = ranker.train(record_dicts)

    return {
        "status": result["status"],
        "total_samples": len(records),
        "features_used": ranker.feature_names,
        "test_r2_score": result.get("r2_score"),
        "message": result["message"],
    }


@router.get("/status")
def get_meta_learning_status(
    db: Session = Depends(get_db),
):
    """Inspect current meta-learning surrogate model state and dataset sample size."""
    ranker = get_meta_ranker()
    record_count = db.query(MetaRunRecord).count()
    return {
        "is_trained": ranker.is_trained,
        "total_historical_runs": record_count,
        "features": ranker.feature_names,
        "status": "trained" if ranker.is_trained else "cold_start_fallback",
    }
