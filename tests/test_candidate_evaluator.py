from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from profiling.profiler import profile_dataset
from recommendations.engine import generate_recommendations
from recommendations.candidate_evaluator import (
    generate_treatment_candidates,
    evaluate_treatment_candidates,
    compute_paired_difference_ci,
    TreatmentCandidate,
    TreatmentValidationReport,
)


@pytest.fixture
def classification_sample_df():
    np.random.seed(42)
    n = 120
    target = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    
    # Feature 1: correlated with target (predictive)
    f1 = target * 2.5 + np.random.normal(0, 0.5, size=n)
    
    # Feature 2: skewed with outliers and missing values
    f2 = np.random.exponential(scale=2.0, size=n)
    f2[np.random.choice(n, size=15, replace=False)] = np.nan
    
    # Feature 3: categorical
    f3 = np.random.choice(["cat_A", "cat_B", "cat_C"], size=n)
    
    # Feature 4: identifier / ID-like
    f4 = [f"USR_{i:04d}" for i in range(n)]

    return pd.DataFrame({
        "pred_feat": f1,
        "skewed_feat": f2,
        "category_feat": f3,
        "user_id": f4,
        "target": target,
    })


@pytest.fixture
def regression_sample_df():
    np.random.seed(42)
    n = 120
    x1 = np.random.normal(20, 5, size=n)
    x2 = np.random.normal(10, 3, size=n)
    x3 = np.random.exponential(1.5, size=n)
    # Target continuous dengan korelasi realistis (di bawah 0.95 agar bukan leakage)
    price = 50 + 1.2 * x1 + 2.0 * x2 - 0.8 * x3 + np.random.normal(0, 4.0, size=n)
    
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "price": price,
    })


def test_generate_treatment_candidates_structure(classification_sample_df):
    profile = profile_dataset(classification_sample_df, target_column="target")
    recs = generate_recommendations(classification_sample_df, profile, task_type="classification")

    candidates = generate_treatment_candidates(profile, recs, task_type="classification")
    assert len(candidates) == 3

    ids = [c.candidate_id for c in candidates]
    assert "recommended" in ids
    assert "baseline" in ids
    assert "alternative" in ids

    rec_cand = next(c for c in candidates if c.candidate_id == "recommended")
    base_cand = next(c for c in candidates if c.candidate_id == "baseline")
    alt_cand = next(c for c in candidates if c.candidate_id == "alternative")

    # Rekomendasi harus mendeteksi drop leakage untuk user_id
    leakage_dec = next((d for d in rec_cand.decisions if d["column_name"] == "user_id"), None)
    if leakage_dec:
        assert leakage_dec["selected_action"] == "drop_column"

    # Verifikasi serialisasi dict
    d = rec_cand.to_dict()
    assert d["candidate_id"] == "recommended"
    assert len(d["decisions"]) > 0
    assert len(d["summary_tags"]) > 0


def test_evaluate_treatment_candidates_classification(classification_sample_df):
    profile = profile_dataset(classification_sample_df, target_column="target")
    recs = generate_recommendations(classification_sample_df, profile, task_type="classification")
    candidates = generate_treatment_candidates(profile, recs, task_type="classification")

    report = evaluate_treatment_candidates(
        df=classification_sample_df,
        profile=profile,
        candidates=candidates,
        target_col="target",
        task_type="classification",
        n_splits=3,
        seed=42,
        confidence_level=0.90,
    )

    assert isinstance(report, TreatmentValidationReport)
    assert report.task_type == "classification"
    assert report.benchmark_model == "Logistic Regression"
    assert report.primary_metric == "F1-Score"
    assert report.validation_status in ("SUPERIOR", "INCONCLUSIVE", "SUBOPTIMAL")
    assert len(report.candidates) == 3

    # Ranking harus 1, 2, 3
    ranks = [c.rank for c in report.candidates]
    assert ranks == [1, 2, 3]

    # Cek nilai metrik valid
    for c in report.candidates:
        assert 0.0 <= c.primary_score <= 1.0
        assert c.training_time_seconds >= 0.0

    # Verifikasi fields paired fold & CI (Requirements 2, 3, 5, 6, 8, 9)
    assert len(report.baseline_fold_scores) == 3
    assert len(report.recommended_fold_scores) == 3
    assert len(report.fold_differences) == 3
    assert report.confidence_level == 0.90
    assert report.ci_lower is not None
    assert report.ci_upper is not None
    assert report.mean_difference is not None
    assert report.relative_difference is not None

    # Verifikasi metadata evaluasi (Requirement 4 & 6)
    meta = report.evaluation_metadata
    assert meta["task_type"] == "classification"
    assert meta["ci_method"] == "paired_two_tailed_student_t"
    assert meta["degrees_of_freedom"] == 2
    assert meta["n_valid_folds"] == 3
    assert meta["estimator"] == "Logistic Regression"
    assert meta["estimator_parameters"]["class_weight"] is None
    assert meta["cv_strategy"] == "StratifiedKFold"
    assert meta["n_splits"] == 3
    assert meta["shuffle"] is True
    assert meta["random_state"] == 42
    assert meta["scoring_metric"] == "f1"
    assert meta["confidence_level"] == 0.90

    # Cek serialisasi dict
    rep_dict = report.to_dict()
    assert "validation_status" in rep_dict
    assert "lift" in rep_dict
    assert "mean_difference" in rep_dict
    assert "ci_lower" in rep_dict
    assert "ci_upper" in rep_dict
    assert "baseline_fold_scores" in rep_dict
    assert "recommended_fold_scores" in rep_dict
    assert "fold_differences" in rep_dict
    assert "evaluation_metadata" in rep_dict
    assert len(rep_dict["candidates"]) == 3


def test_evaluate_treatment_candidates_regression(regression_sample_df):
    profile = profile_dataset(regression_sample_df, target_column="price")
    recs = generate_recommendations(regression_sample_df, profile, task_type="regression")
    candidates = generate_treatment_candidates(profile, recs, task_type="regression")

    report = evaluate_treatment_candidates(
        df=regression_sample_df,
        profile=profile,
        candidates=candidates,
        target_col="price",
        task_type="regression",
        n_splits=3,
        seed=42,
    )

    assert report.task_type == "regression"
    assert report.benchmark_model == "Ridge Regression"
    assert report.primary_metric == "R² Score"
    assert report.validation_status in ("SUPERIOR", "INCONCLUSIVE", "SUBOPTIMAL")
    assert len(report.candidates) == 3

    # Pada data regresi linear ini, model harus menghasilkan R2 tinggi
    best_cand = report.candidates[0]
    assert best_cand.primary_score > 0.5
    assert best_cand.secondary_metric_name == "RMSE"

    # Verifikasi metadata evaluasi regresi
    meta = report.evaluation_metadata
    assert meta["estimator"] == "Ridge Regression"
    assert meta["cv_strategy"] == "KFold"
    assert meta["scoring_metric"] == "r2"


def test_evaluate_treatment_candidates_missing_target_error(classification_sample_df):
    profile = profile_dataset(classification_sample_df, target_column="target")
    candidates = generate_treatment_candidates(profile, [], task_type="classification")

    with pytest.raises(ValueError, match="Target column 'non_existent' not found"):
        evaluate_treatment_candidates(
            df=classification_sample_df,
            profile=profile,
            candidates=candidates,
            target_col="non_existent",
            task_type="classification",
        )


# =============================================================================
# UNIT TESTS FOR STATISTICAL ENGINE & EDGE CASES (Requirements 3, 4, 7, 8, 9)
# =============================================================================

def test_compute_paired_difference_ci_superior():
    # Seluruh fold recommended konsisten lebih unggul dari baseline
    rec = [0.85, 0.88, 0.90]
    base = [0.70, 0.72, 0.75]
    result = compute_paired_difference_ci(rec, base, confidence_level=0.90)

    assert result["validation_status"] == "SUPERIOR"
    assert result["ci_lower"] > 0
    assert result["ci_upper"] > result["ci_lower"]
    assert result["mean_difference"] > 0
    assert "3/3 fold positif" in result["sign_agreement"]
    assert "Terindikasi Unggul" in result["headline"]


def test_compute_paired_difference_ci_suboptimal():
    # Baseline konsisten lebih unggul dari recommended
    rec = [0.60, 0.65, 0.62]
    base = [0.80, 0.82, 0.85]
    result = compute_paired_difference_ci(rec, base, confidence_level=0.90)

    assert result["validation_status"] == "SUBOPTIMAL"
    assert result["ci_upper"] < 0
    assert result["mean_difference"] < 0
    assert "0/3 fold positif" in result["sign_agreement"]
    assert "Terindikasi Suboptimal" in result["headline"]


def test_compute_paired_difference_ci_inconclusive_crosses_zero():
    # Selisih antar fold fluktuatif (ada positif, ada negatif) sehingga CI memotong nol
    rec = [0.82, 0.74, 0.86]
    base = [0.80, 0.78, 0.84]
    result = compute_paired_difference_ci(rec, base, confidence_level=0.90)

    assert result["validation_status"] == "INCONCLUSIVE"
    assert result["ci_lower"] < 0 < result["ci_upper"]
    assert "Inconclusive" in result["headline"]


def test_compute_paired_difference_ci_edge_case_insufficient_folds():
    # k < 2 (hanya 1 fold)
    result = compute_paired_difference_ci([0.8], [0.7], confidence_level=0.90)
    assert result["validation_status"] == "INVALID_EVALUATION"
    assert result["ci_lower"] is None
    assert result["ci_upper"] is None
    assert "Jumlah fold kurang dari 2" in result["headline"]


def test_compute_paired_difference_ci_edge_case_nan_or_inf():
    # Terdapat nilai NaN pada fold
    rec = [0.85, float("nan"), 0.90]
    base = [0.70, 0.72, 0.75]
    result = compute_paired_difference_ci(rec, base, confidence_level=0.90)

    assert result["validation_status"] == "INVALID_EVALUATION"
    assert result["ci_lower"] is None
    assert result["ci_upper"] is None
    assert "NaN atau tak hingga" in result["headline"]


def test_compute_paired_difference_ci_edge_case_zero_variance():
    # Perbedaan identik persis di seluruh fold (std dev diff = 0)
    rec = [0.85, 0.87, 0.89]
    base = [0.80, 0.82, 0.84]  # diffs = [0.05, 0.05, 0.05]
    result = compute_paired_difference_ci(rec, base, confidence_level=0.90)

    assert result["validation_status"] == "SUPERIOR"
    assert pytest.approx(result["ci_lower"], 1e-4) == 0.05
    assert pytest.approx(result["ci_upper"], 1e-4) == 0.05
    assert pytest.approx(result["mean_difference"], 1e-4) == 0.05


def test_compute_paired_difference_ci_configurable_confidence():
    rec = [0.85, 0.87, 0.90]
    base = [0.82, 0.80, 0.86]
    r90 = compute_paired_difference_ci(rec, base, confidence_level=0.90)
    r95 = compute_paired_difference_ci(rec, base, confidence_level=0.95)

    # Margin 95% CI harus lebih lebar dibanding 90% CI
    width_90 = r90["ci_upper"] - r90["ci_lower"]
    width_95 = r95["ci_upper"] - r95["ci_lower"]
    assert width_95 > width_90


def test_compute_paired_difference_ci_guarded_relative_difference_zero_baseline():
    # Ketika baseline fold semuanya 0.0 (misal: F1 = 0 karena tidak ada minority class terprediksi)
    rec = [0.25, 0.30, 0.20]
    base = [0.0, 0.0, 0.0]
    result = compute_paired_difference_ci(rec, base, confidence_level=0.90)

    assert result["validation_status"] == "SUPERIOR"
    assert result["mean_difference"] > 0
    # Guard: relative_difference harus None, bukan division by zero atau inf atau 0.0 semu
    assert result["relative_difference"] is None
    assert "rel: N/A" in result["headline"]

