from __future__ import annotations

import logging
from typing import Any, Callable

from jobs.task_registry import register_task

logger = logging.getLogger("jobs.handlers")


@register_task("echo")
def handle_echo(payload: dict[str, Any], update_progress: Callable[[float], None]) -> dict[str, Any]:
    """Simple test task that echos back payload with progress ticks."""
    update_progress(25.0)
    msg = payload.get("message", "hello")
    update_progress(75.0)
    return {"status": "ok", "echo": msg, "received_payload": payload}


@register_task("meta_learning_train")
def handle_meta_learning_train(payload: dict[str, Any], update_progress: Callable[[float], None]) -> dict[str, Any]:
    """Trains the surrogate meta-learning ranker model."""
    from recommendations.meta_ranker import get_meta_ranker

    update_progress(10.0)
    ranker = get_meta_ranker()
    records = payload.get("records", [])
    update_progress(50.0)
    result = ranker.train(records)
    update_progress(90.0)
    return result


@register_task("evaluate_candidates")
def handle_evaluate_candidates(payload: dict[str, Any], update_progress: Callable[[float], None]) -> dict[str, Any]:
    """Background candidate treatment strategy evaluation."""
    import pandas as pd
    from recommendations.candidate_evaluator import (
        evaluate_treatment_candidates,
        generate_treatment_candidates,
    )

    update_progress(20.0)
    profile = payload.get("profile", {})
    task_type = payload.get("task_type", "classification")
    target_col = payload.get("target_column")
    raw_data = payload.get("data")

    if not raw_data or not target_col:
        raise ValueError("Missing data or target_column in payload")

    df = pd.DataFrame(raw_data)
    candidates = generate_treatment_candidates(profile, task_type, target_col)
    update_progress(50.0)

    report = evaluate_treatment_candidates(
        df=df,
        target_col=target_col,
        task_type=task_type,
        candidates=candidates,
        n_splits=3,
        random_state=42,
    )
    update_progress(90.0)
    return report.model_dump()
