from __future__ import annotations

import sys
from pathlib import Path

# Add packages/ml_engine to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "ml_engine"))

from assistant.explainer import explain_recommendation, summarize_experiments, suggest_next_step
from assistant.llm_provider import is_llm_available, generate_llm_enrichment


def test_explain_recommendation_missing_value_median():
    rec = {
        "category": "missing_values",
        "column_name": "age",
        "recommendation": "median_imputation",
        "score": 90,
        "evidence": {"missing_count": 10, "missing_rate": 0.1, "skewness": 2.5},
        "rationale": "High skewness requires median imputation.",
        "alternative_actions": ["mean_imputation"],
    }
    exp = explain_recommendation(rec)

    assert "age" in exp["headline"]
    assert "Median" in exp["headline"]
    assert exp["confidence_score"] == 90
    assert "outlier" in exp["detailed_explanation"]
    assert len(exp["key_evidence"]) > 0


def test_explain_recommendation_real_evidence_strings():
    """Test with real-world string evidence produced by recommendations engine."""
    rec = {
        "category": "missing_value",
        "column_name": "income",
        "recommendation": "median_imputation",
        "score": "80",
        "evidence": {
            "missing_rate": "12.5%",
            "dtype": "numeric",
            "skewness": "1.85",
            "sample_size": "200",
        },
        "rationale": "Skewed feature with missing values.",
        "alternative_actions": ["mean_imputation", "drop_column"],
    }
    exp = explain_recommendation(rec)

    assert "income" in exp["headline"]
    assert "25 baris (12.5%)" in exp["detailed_explanation"]
    assert "1.85" in exp["detailed_explanation"]
    assert "Median" in exp["detailed_explanation"]


def test_explain_recommendation_scaling_standard():
    rec = {
        "category": "scaling",
        "column_name": "income",
        "recommendation": "standard_scaling",
        "score": 85,
        "evidence": {"skewness": 0.1},
        "rationale": "Near normal distribution.",
    }
    exp = explain_recommendation(rec)

    assert "Standard Scaling" in exp["action_title"]
    assert "income" in exp["headline"]


def test_explain_recommendation_data_leakage_proxy():
    rec = {
        "category": "data_leakage",
        "column_name": "leaked_feature",
        "recommendation": "drop_column",
        "score": 95,
        "evidence": {"leakage_type": "target_proxy", "correlation_value": "0.992"},
        "rationale": "High correlation with target.",
    }
    exp = explain_recommendation(rec)

    assert "Drop Column" in exp["action_title"]
    assert "leaked_feature" in exp["headline"]
    assert "Target Proxy" in exp["detailed_explanation"]


def test_explain_recommendation_data_leakage_identifier():
    rec = {
        "category": "data_leakage",
        "column_name": "user_id",
        "recommendation": "drop_column",
        "score": 95,
        "evidence": {"leakage_type": "identifier", "unique_ratio": "1.0"},
        "rationale": "Unique identifier column.",
    }
    exp = explain_recommendation(rec)

    assert "Drop Column" in exp["action_title"]
    assert "user_id" in exp["headline"]
    assert "Identifier" in exp["detailed_explanation"]


def test_summarize_experiments_finds_best_classification():
    exps = [
        {
            "model_name": "Logistic Regression",
            "status": "completed",
            "training_time_seconds": 0.5,
            "metrics": [{"name": "f1_mean", "value": 0.85, "split": "cv"}, {"name": "accuracy_mean", "value": 0.86, "split": "cv"}],
        },
        {
            "model_name": "Random Forest",
            "status": "completed",
            "training_time_seconds": 1.2,
            "metrics": [{"name": "f1_mean", "value": 0.94, "split": "cv"}, {"name": "accuracy_mean", "value": 0.95, "split": "cv"}],
        },
    ]

    summary = summarize_experiments(
        experiments=exps,
        candidate={"model_name": "Random Forest"},
        target_column="target",
        task_type="classification",
    )

    assert summary["best_model_name"] == "Random Forest"
    assert "94.00%" in summary["headline"]
    assert summary["best_metric_name"] == "F1-Score"
    assert len(summary["strengths"]) > 0


def test_summarize_experiments_regression():
    """Verify regression summarization correctly ranks by R2 score."""
    exps = [
        {
            "model_name": "Linear Regression",
            "status": "completed",
            "training_time_seconds": 0.2,
            "metrics": [
                {"name": "r2_mean", "value": 0.78, "split": "cv"},
                {"name": "rmse_mean", "value": 4.5, "split": "cv"},
                {"name": "mae_mean", "value": 3.2, "split": "cv"},
            ],
        },
        {
            "model_name": "Random Forest Regressor",
            "status": "completed",
            "training_time_seconds": 1.1,
            "metrics": [
                {"name": "r2_mean", "value": 0.925, "split": "cv"},
                {"name": "rmse_mean", "value": 2.1, "split": "cv"},
                {"name": "mae_mean", "value": 1.5, "split": "cv"},
            ],
        },
    ]

    summary = summarize_experiments(
        experiments=exps,
        candidate={"model_name": "Random Forest Regressor"},
        target_column="house_price",
        task_type="regression",
    )

    assert summary["best_model_name"] == "Random Forest Regressor"
    assert "0.9250" in summary["headline"]
    assert summary["best_metric_name"] == "R²"
    assert "2.1000" in summary["narrative"]
    assert len(summary["strengths"]) > 0


def test_summarize_experiments_deduplicates_runner_up_and_formats_display_name():
    """Verify that multiple runs of the same algorithm do not declare the same model as runner-up with 0 diff."""
    exps = [
        {
            "model_name": "gradient_boosting_regressor",
            "status": "completed",
            "training_time_seconds": 1.2,
            "metrics": [
                {"name": "r2_mean", "value": 0.8273, "split": "cv"},
                {"name": "rmse_mean", "value": 218589.8038, "split": "cv"},
                {"name": "mae_mean", "value": 27049.6202, "split": "cv"},
            ],
        },
        {
            "model_name": "gradient_boosting_regressor",
            "status": "completed",
            "training_time_seconds": 1.1,
            "metrics": [
                {"name": "r2_mean", "value": 0.8273, "split": "cv"},
                {"name": "rmse_mean", "value": 218589.8038, "split": "cv"},
                {"name": "mae_mean", "value": 27049.6202, "split": "cv"},
            ],
        },
        {
            "model_name": "linear_regression",
            "status": "completed",
            "training_time_seconds": 0.3,
            "metrics": [
                {"name": "r2_mean", "value": 0.6521, "split": "cv"},
                {"name": "rmse_mean", "value": 310000.0, "split": "cv"},
                {"name": "mae_mean", "value": 45000.0, "split": "cv"},
            ],
        },
    ]

    summary = summarize_experiments(
        experiments=exps,
        candidate={"model_name": "gradient_boosting_regressor"},
        target_column="price",
        task_type="regression",
    )

    # Nama model harus terformat rapi dan profesional
    assert summary["best_model_name"] == "Gradient Boosting Regressor"
    assert "Gradient Boosting Regressor" in summary["headline"]
    assert "selisih 0.0000" not in summary["narrative"]
    assert "Linear Regression" in summary["narrative"]
    assert "218,589.8038" in summary["narrative"]
    assert "Gradient Boosting Regressor" in summary["takeaways"][1]


def test_suggest_next_step():
    step1 = suggest_next_step(current_page="upload", has_dataset=False)
    assert "Unggah" in step1["step_title"]

    step2 = suggest_next_step(current_page="upload", has_dataset=True, has_decisions=False)
    assert "EDA" in step2["step_title"]
    assert step2["next_url"] == "/eda"

    step3 = suggest_next_step(current_page="eda", has_dataset=True, has_decisions=False)
    assert "Treatment" in step3["action_label"]
    assert step3["next_url"] == "/treatment"

    step5 = suggest_next_step(current_page="experiments", has_dataset=True, has_decisions=True, has_pipeline=True, has_experiments=True, candidate_selected=False)
    assert "Kandidat" in step5["step_title"]


def test_explain_recommendation_singular_categories_and_null_skewness():
    rec_missing = {
        "category": "missing_value",
        "column_name": "age",
        "recommendation": "median_imputation",
        "score": 90,
        "evidence": {"missing_count": 10, "missing_rate": 0.1, "skewness": None},
        "rationale": "Missing value handling.",
    }
    exp_missing = explain_recommendation(rec_missing)
    assert "age" in exp_missing["detailed_explanation"]
    assert "skewness = tinggi" in exp_missing["detailed_explanation"]

    rec_outlier = {
        "category": "outlier",
        "column_name": "salary",
        "recommendation": "winsorization",
        "score": 85,
        "evidence": {"outlier_count": 5, "outlier_rate": 0.02},
    }
    exp_outlier = explain_recommendation(rec_outlier)
    assert "salary" in exp_outlier["detailed_explanation"]
    assert "Winsorization" in exp_outlier["detailed_explanation"]

    rec_imbalance = {
        "category": "class_imbalance",
        "column_name": "churn",
        "recommendation": "stratified_eval",
        "score": 95,
        "evidence": {"imbalance_ratio": 4.5, "minority_pct": 0.18},
    }
    exp_imbalance = explain_recommendation(rec_imbalance)
    assert "churn" in exp_imbalance["detailed_explanation"]
    assert "4.5:1" in exp_imbalance["detailed_explanation"]


def test_llm_provider_fallback():
    # Should safely return boolean and handle absence of LLM keys
    avail = is_llm_available()
    assert isinstance(avail, bool)

    enrichment = generate_llm_enrichment("test prompt")
    # Will return None if no GEMINI_API_KEY set
    assert enrichment is None or isinstance(enrichment, str)
