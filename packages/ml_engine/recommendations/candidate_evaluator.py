from __future__ import annotations

"""
=============================================================================
MODUL: CANDIDATE TREATMENT GENERATION & EMPIRICAL VALIDATION
(packages/ml_engine/recommendations/candidate_evaluator.py)
=============================================================================
Deskripsi:
Modul ini bertanggung jawab merakit varian paket strategi pra-pemrosesan data
(Treatment Candidates) dan mengujinya secara objektif menggunakan Cross-Validation
dengan model patokan cepat (Fast Reference Benchmark Estimator).

Prinsip Utama (Roadmap V2 Section 0.5 & 7.2):
- Detect Issues -> Generate Candidate Pipelines -> Evaluate via CV -> Rank -> Validate
- Bebas kebocoran data (fit transformer hanya terjadi pada training fold CV).
- Menghasilkan Treatment Lift empiris untuk membuktikan efektivitas rekomendasi.
=============================================================================
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from evaluation.evaluator import run_cross_validation
from preprocessing.pipeline_builder import build_pipeline
from profiling.profiler import DatasetProfile
from recommendations.engine import Recommendation


@dataclass
class TreatmentCandidate:
    """
    Representasi satu paket strategi pra-pemrosesan data (Treatment Candidate).
    """
    candidate_id: str                      # "recommended" | "baseline" | "alternative"
    name: str                              # Nama strategi yang ramah pengguna
    strategy_type: str                     # "recommended" | "baseline" | "alternative"
    description: str                       # Penjelasan ringkas strategi
    decisions: list[dict[str, Any]]        # Daftar keputusan per-kolom untuk pipeline builder
    summary_tags: list[str]                # Tag ringkasan tindakan (misal: ["Median Impute", "Robust Scale"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TreatmentCandidateEvaluation:
    """
    Hasil evaluasi Cross-Validation untuk satu kandidat treatment.
    """
    candidate_id: str
    name: str
    strategy_type: str
    primary_metric_name: str
    primary_score: float                   # Nilai rata-rata metrik utama (misal: f1_mean atau r2_mean)
    primary_std: float                     # Standar deviasi antar fold
    secondary_metric_name: str
    secondary_score: float                 # Metrik pelengkap (misal: accuracy_mean atau rmse_mean)
    rank: int
    summary_tags: list[str]
    training_time_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_round(val: float | None, digits: int = 4) -> float | None:
    if val is None:
        return None
    try:
        fval = float(val)
        if np.isnan(fval) or np.isinf(fval):
            return None
        return round(fval, digits)
    except (ValueError, TypeError):
        return None


@dataclass
class TreatmentValidationReport:
    """
    Laporan perbandingan dan validasi empiris seluruh kandidat treatment
    menggunakan Fast Proxy Evaluation berbasis paired difference CI.
    """
    task_type: str                         # "classification" | "regression"
    benchmark_model: str                   # Nama model pembanding (Logistic Regression / Ridge)
    primary_metric: str                    # "F1-Score" | "R² Score"
    validation_status: str                 # "SUPERIOR" | "INCONCLUSIVE" | "SUBOPTIMAL" | "INVALID_EVALUATION"
    headline: str                          # Kalimat ringkas status validasi proksi
    explanation: str                       # Uraian alasan berbasis paired differences dan CI
    lift: float | None                     # mean_difference: Score(recommended) - Score(baseline)
    lift_percentage: float | None          # relative_difference: (lift / baseline) * 100
    best_candidate_id: str                 # ID kandidat peraih skor tertinggi
    candidates: list[TreatmentCandidateEvaluation]
    # Auditable metrics & Paired Fold statistics
    mean_difference: float | None = 0.0
    relative_difference: float | None = 0.0
    confidence_level: float = 0.90
    ci_lower: float | None = 0.0
    ci_upper: float | None = 0.0
    baseline_fold_scores: list[float] = field(default_factory=list)
    recommended_fold_scores: list[float] = field(default_factory=list)
    fold_differences: list[float] = field(default_factory=list)
    sign_agreement: str = ""
    evaluation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "benchmark_model": self.benchmark_model,
            "primary_metric": self.primary_metric,
            "validation_status": self.validation_status,
            "headline": self.headline,
            "explanation": self.explanation,
            "lift": _safe_round(self.lift, 4),
            "lift_percentage": _safe_round(self.lift_percentage, 2),
            "mean_difference": _safe_round(self.mean_difference, 4),
            "relative_difference": _safe_round(self.relative_difference, 2),
            "best_candidate_id": self.best_candidate_id,
            "confidence_level": self.confidence_level,
            "ci_lower": _safe_round(self.ci_lower, 4),
            "ci_upper": _safe_round(self.ci_upper, 4),
            "baseline_fold_scores": [_safe_round(s, 4) for s in self.baseline_fold_scores if s is not None],
            "recommended_fold_scores": [_safe_round(s, 4) for s in self.recommended_fold_scores if s is not None],
            "fold_differences": [_safe_round(d, 4) for d in self.fold_differences if d is not None],
            "sign_agreement": self.sign_agreement,
            "evaluation_metadata": self.evaluation_metadata,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# =============================================================================
# 1. CANDIDATE TREATMENT GENERATOR
# =============================================================================

def generate_treatment_candidates(
    profile: DatasetProfile,
    recommendations: list[Recommendation],
    task_type: str = "classification",
) -> list[TreatmentCandidate]:
    """
    Menyusun 3 paket strategi perlakuan data yang rasional dan siap diadu:
    1. Recommended: Mengikuti seluruh rekomendasi berbasis bukti statistik.
    2. Baseline: Preprocessing standar minimal tanpa transformasi tingkat lanjut.
    3. Alternative: Memanfaatkan opsi alternatif tindakan dari rekomendasi.
    """
    rec_decisions: list[dict[str, Any]] = []
    base_decisions: list[dict[str, Any]] = []
    alt_decisions: list[dict[str, Any]] = []

    rec_tags: set[str] = set()
    base_tags: set[str] = {"Mean Imputation", "Standard Scaling", "One-Hot Encoding"}
    alt_tags: set[str] = set()

    for rec in recommendations:
        col = rec.column
        cat = rec.category
        primary_action = rec.recommended_action
        alts = rec.alternatives or []

        # 1. Keputusan Kandidat Rekomendasi
        rec_decisions.append({
            "column_name": col,
            "category": cat,
            "selected_action": primary_action,
            "status": "accepted",
        })

        if cat == "data_leakage" and primary_action == "drop_column":
            rec_tags.add("Drop Leakage Column")
        elif cat == "missing_value":
            rec_tags.add(f"{primary_action.replace('_', ' ').title()}")
        elif cat == "scaling":
            rec_tags.add(f"{primary_action.replace('_', ' ').title()}")
        elif cat == "encoding":
            rec_tags.add(f"{primary_action.replace('_', ' ').title()}")
        elif cat == "outlier" and primary_action == "winsorization":
            rec_tags.add("Winsorizer Outliers")

        # 2. Keputusan Kandidat Baseline (Minimal / Default)
        # Catatan integritas data: Kolom leakage ber-severity tinggi tetap di-drop
        # agar baseline tidak curang memprediksi target lewat proxy leakage.
        if cat == "data_leakage" and primary_action == "drop_column":
            base_action = "drop_column"
            base_tags.add("Drop Leakage Column")
        elif cat == "missing_value":
            base_action = "mean_imputation"
        elif cat == "scaling":
            base_action = "standard_scaling"
        elif cat == "encoding":
            base_action = "one_hot_encoding"
        elif cat == "outlier":
            base_action = "leave_unchanged"
        elif cat in ("target_transformation", "target_outliers"):
            base_action = "leave_unchanged"
        else:
            base_action = primary_action

        base_decisions.append({
            "column_name": col,
            "category": cat,
            "selected_action": base_action,
            "status": "accepted",
        })

        # 3. Keputusan Kandidat Alternatif (Robust / Variasi)
        if cat == "data_leakage" and primary_action == "drop_column":
            alt_action = "drop_column"
            alt_tags.add("Drop Leakage Column")
        elif alts:
            alt_action = alts[0]
            alt_tags.add(f"{alt_action.replace('_', ' ').title()}")
        else:
            alt_action = primary_action

        alt_decisions.append({
            "column_name": col,
            "category": cat,
            "selected_action": alt_action,
            "status": "accepted",
        })

    # Paket 1: Rekomendasi Utama
    candidate_recommended = TreatmentCandidate(
        candidate_id="recommended",
        name="Strategi Rekomendasi (Evidence-Based)",
        strategy_type="recommended",
        description="Menerapkan seluruh rekomendasi berbasis bukti statistik profil dataset (penanganan skewness, outlier robust, dan eliminasi kebocoran data).",
        decisions=rec_decisions,
        summary_tags=sorted(list(rec_tags)) if rec_tags else ["Evidence-Driven Preprocessing"],
    )

    # Paket 2: Baseline Minimal
    candidate_baseline = TreatmentCandidate(
        candidate_id="baseline",
        name="Strategi Baseline (Minimal / Standar)",
        strategy_type="baseline",
        description="Pembersihan standar sederhana: imputasi nilai rata-rata dan penskalaan standar tanpa perlakuan distribusi khusus.",
        decisions=base_decisions,
        summary_tags=sorted(list(base_tags)),
    )

    # Paket 3: Alternatif / Robust
    candidate_alternative = TreatmentCandidate(
        candidate_id="alternative",
        name="Strategi Alternatif (Robust / Variasi)",
        strategy_type="alternative",
        description="Menerapkan opsi tindakan alternatif yang tercantum pada rekomendasi (misal: penanganan outlier, min-max scaling, frequency encoding).",
        decisions=alt_decisions,
        summary_tags=sorted(list(alt_tags)) if alt_tags else ["Alternative Encoders & Scalers"],
    )

    return [candidate_recommended, candidate_baseline, candidate_alternative]


# =============================================================================
# 2. STATISTICAL ENGINE: PAIRED-FOLD CI SCREENING
# =============================================================================

def compute_paired_difference_ci(
    rec_scores: list[float],
    base_scores: list[float],
    confidence_level: float = 0.90,
    cv_desc: str = "3-Fold CV",
    benchmark_model: str = "Logistic Regression",
) -> dict[str, Any]:
    """
    Menghitung paired-fold CI screening berbasis selisih antar-fold berpasangan
    dan interval kepercayaan dua sisi Student's t untuk Fast Proxy Evaluation.

    Prinsip Metodologis (Roadmap V2 Section 0.5 & 7.2):
    - Unit observasi adalah selisih fold berpasangan: d_i = rec_score_i - base_score_i.
    - Standard Error (SE) = s_d / sqrt(k) dengan derajat kebebasan df = k - 1.
    - Two-tailed Student's t confidence interval: CI = mean(d) +/- t_crit * SE.
    - Dasar Keputusan:
        * CI_lower > 0 -> SUPERIOR (indikasi empiris keunggulan di bawah screening proksi)
        * CI_upper < 0 -> SUBOPTIMAL (indikasi empiris bahwa baseline lebih baik di screening proksi)
        * 0 in [CI_lower, CI_upper] -> INCONCLUSIVE (perbedaan dalam batas variasi sampling fold)
    - Sign agreement (N_pos / k) disimpan sebagai bukti pendukung, BUKAN pemicu otomatis.
    - Wording menyatakan hasil secara presisi sebagai paired-fold CI screening dalam protokol proksi CV.
    """
    # 1. Edge Case: Validasi kecukupan dan integritas fold
    if not rec_scores or not base_scores or len(rec_scores) != len(base_scores) or len(rec_scores) < 2:
        mean_d = float(np.mean(rec_scores) - np.mean(base_scores)) if (rec_scores and base_scores and len(rec_scores) == len(base_scores)) else None
        mean_b = float(np.mean(base_scores)) if base_scores else None
        # Guard: hindari pembagian nol saat baseline = 0
        rel_d = (mean_d / abs(mean_b) * 100.0) if (mean_d is not None and mean_b is not None and abs(mean_b) > 1e-6 and mean_b > 0) else None
        return {
            "validation_status": "INVALID_EVALUATION",
            "headline": "Evaluasi Proksi Tidak Valid: Jumlah fold kurang dari 2",
            "explanation": "Evaluasi memerlukan minimal 2 fold valid untuk menghitung paired-fold Student's t confidence interval.",
            "mean_difference": mean_d,
            "relative_difference": rel_d,
            "ci_lower": None,
            "ci_upper": None,
            "sign_agreement": "",
            "ci_method": "paired_two_tailed_student_t",
            "degrees_of_freedom": None,
            "n_valid_folds": len(rec_scores) if rec_scores else 0,
        }

    # 2. Edge Case: Nilai NaN atau tak hingga
    rec_arr = np.array(rec_scores, dtype=float)
    base_arr = np.array(base_scores, dtype=float)
    if not np.all(np.isfinite(rec_arr)) or not np.all(np.isfinite(base_arr)):
        return {
            "validation_status": "INVALID_EVALUATION",
            "headline": "Evaluasi Proksi Tidak Valid: Terdapat nilai fold NaN atau tak hingga",
            "explanation": "Perhitungan metrik pada salah satu fold menghasilkan NaN atau nilai tak hingga, sehingga paired-fold CI screening tidak valid.",
            "mean_difference": None,
            "relative_difference": None,
            "ci_lower": None,
            "ci_upper": None,
            "sign_agreement": "",
            "ci_method": "paired_two_tailed_student_t",
            "degrees_of_freedom": None,
            "n_valid_folds": 0,
        }

    # 3. Hitung selisih berpasangan per fold
    d = rec_arr - base_arr
    k = len(d)
    mean_d = float(np.mean(d))
    mean_base = float(np.mean(base_arr))

    # Guard: pembagian nol ketika baseline = 0 (misal: F1 = 0)
    if abs(mean_base) > 1e-6 and mean_base > 0:
        rel_d = (mean_d / abs(mean_base)) * 100.0
    else:
        rel_d = None

    n_pos = int(np.sum(d > 1e-9))
    sign_agreement = f"{n_pos}/{k} fold positif"
    conf_pct = int(round(confidence_level * 100))

    # 4. Edge Case: Varians selisih nol (seluruh fold menghasilkan selisih identik)
    s_d = float(np.std(d, ddof=1))
    if s_d < 1e-12:
        ci_lower = mean_d
        ci_upper = mean_d
        if mean_d > 1e-6:
            status = "SUPERIOR"
            headline = f"Strategi Rekomendasi Terindikasi Unggul ({mean_d:+.4f})"
            explanation = (
                f"Berdasarkan paired-fold CI screening ({cv_desc}) dengan model proksi ({benchmark_model}), "
                f"seluruh fold menghasilkan peningkatan skor konstan ({mean_d:+.4f}, varians fold = 0). "
                f"Hasil screening ini memberikan indikasi awal keunggulan strategi rekomendasi pada model proksi (sign agreement: {sign_agreement})."
            )
        elif mean_d < -1e-6:
            status = "SUBOPTIMAL"
            headline = f"Strategi Rekomendasi Terindikasi Suboptimal pada Model Proksi ({mean_d:+.4f})"
            explanation = (
                f"Berdasarkan paired-fold CI screening ({cv_desc}) dengan model proksi ({benchmark_model}), "
                f"baseline mencatat skor lebih tinggi konstan di seluruh fold ({mean_d:+.4f}, varians fold = 0). "
                f"Ini menunjukkan strategi rekomendasi belum memberikan keuntungan pada model proksi linier (sign agreement: {sign_agreement})."
            )
        else:
            status = "INCONCLUSIVE"
            headline = f"Performa Identik dengan Baseline ({mean_d:+.4f})"
            explanation = (
                f"Berdasarkan paired-fold CI screening ({cv_desc}) dengan model proksi ({benchmark_model}), "
                f"skor rekomendasi dan baseline bernilai persis sama di setiap fold (perbedaan = 0.0000). "
                f"Tidak ditemukan perbedaan performa pada model proksi."
            )
        return {
            "validation_status": status,
            "headline": headline,
            "explanation": explanation,
            "mean_difference": mean_d,
            "relative_difference": rel_d,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "sign_agreement": sign_agreement,
            "ci_method": "paired_two_tailed_student_t",
            "degrees_of_freedom": k - 1,
            "n_valid_folds": k,
        }

    # 5. Kasus Standar: Student's t distribution dengan df = k - 1
    se = s_d / np.sqrt(k)
    df = k - 1
    t_crit = float(stats.t.ppf((1.0 + confidence_level) / 2.0, df=df))
    margin = t_crit * se
    ci_lower = float(mean_d - margin)
    ci_upper = float(mean_d + margin)

    diff_str = f"{mean_d:+.4f} ({rel_d:+.1f}%)" if rel_d is not None else f"{mean_d:+.4f} (rel: N/A)"
    ci_str = f"[{ci_lower:+.4f}, {ci_upper:+.4f}]"

    if ci_lower > 0:
        status = "SUPERIOR"
        headline = f"Strategi Rekomendasi Terindikasi Unggul ({diff_str})"
        explanation = (
            f"Berdasarkan paired-fold CI screening ({cv_desc}) dengan model proksi ({benchmark_model}), "
            f"interval kepercayaan perbedaan berpasangan {conf_pct}% CI {ci_str} seluruhnya berada di atas nol. "
            f"Hasil screening ini memberikan indikasi empiris bahwa strategi rekomendasi mencatat skor lebih tinggi "
            f"pada model proksi dibanding baseline (sign agreement: {sign_agreement}). "
            f"Catatan: Screening ini adalah evaluasi cepat terisolasi pada model linier sederhana; karakteristik pada model ensemble "
            f"non-linear (seperti Random Forest atau Gradient Boosting) dapat bervariasi pada eksperimen penuh."
        )
    elif ci_upper < 0:
        status = "SUBOPTIMAL"
        headline = f"Strategi Rekomendasi Terindikasi Suboptimal pada Model Proksi ({diff_str})"
        explanation = (
            f"Berdasarkan paired-fold CI screening ({cv_desc}) dengan model proksi ({benchmark_model}), "
            f"strategi baseline mencatat skor lebih tinggi dengan interval kepercayaan perbedaan berpasangan {conf_pct}% CI {ci_str} "
            f"seluruhnya berada di bawah nol (sign agreement: {sign_agreement}). Ini memberikan indikasi empiris bahwa transformasi "
            f"rekomendasi belum memberikan keuntungan pada model proksi linier. Perhatikan bahwa model non-linear mungkin merespons secara berbeda pada eksperimen penuh."
        )
    else:
        status = "INCONCLUSIVE"
        headline = f"Perbedaan Belum Meyakinkan / Inconclusive ({diff_str})"
        explanation = (
            f"Berdasarkan paired-fold CI screening ({cv_desc}) dengan model proksi ({benchmark_model}), "
            f"interval kepercayaan perbedaan berpasangan {conf_pct}% CI {ci_str} mencakup angka nol. "
            f"Hal ini mengindikasikan bahwa perbedaan skor rata-rata ({diff_str}) masih berada dalam rentang variasi sampling "
            f"antar-fold proksi (sign agreement: {sign_agreement}), bukan bukti keunggulan atau kelemahan yang definitif. "
            f"Transformasi rekomendasi (seperti outlier handling dan scaling) tetap relevan dipertimbangkan untuk model downstream yang sensitif terhadap anomali data."
        )

    return {
        "validation_status": status,
        "headline": headline,
        "explanation": explanation,
        "mean_difference": mean_d,
        "relative_difference": rel_d,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "sign_agreement": sign_agreement,
        "ci_method": "paired_two_tailed_student_t",
        "degrees_of_freedom": df,
        "n_valid_folds": k,
    }


# =============================================================================
# 3. CANDIDATE TREATMENT EVALUATOR & BENCHMARK RUNNER
# =============================================================================

def evaluate_treatment_candidates(
    df: pd.DataFrame,
    profile: DatasetProfile,
    candidates: list[TreatmentCandidate],
    target_col: str,
    task_type: str = "classification",
    n_splits: int = 3,
    seed: int = 42,
    confidence_level: float = 0.90,
) -> TreatmentValidationReport:
    """
    Mengevaluasi seluruh Treatment Candidates secara objektif dengan K-Fold CV
    menggunakan model patokan deterministik cepat (Fast Reference Estimator)
    dan paired difference Student's t confidence interval.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Pisahkan fitur dan label target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    is_regression = task_type == "regression"

    if is_regression:
        y = pd.to_numeric(y, errors="coerce")
        # Bersihkan baris di mana target bernilai NaN
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        benchmark_estimator = Ridge(random_state=seed)
        benchmark_name = "Ridge Regression"
        primary_metric_key = "r2_mean"
        primary_metric_label = "R² Score"
        primary_metric_raw = "r2"
        secondary_metric_key = "rmse_mean"
        secondary_metric_label = "RMSE"

        # Pre-generate CV splits untuk paired comparison
        cv_strategy_name = "KFold"
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pre_generated_splits = list(splitter.split(X, y))
    else:
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        # Requirement 1: Keep class_weight=None for Proxy Evaluator
        # dan tidak menggunakan heuristik imbalance
        benchmark_estimator = LogisticRegression(
            max_iter=500,
            class_weight=None,
            random_state=seed,
        )
        benchmark_name = "Logistic Regression"
        primary_metric_key = "f1_mean"
        primary_metric_label = "F1-Score"
        primary_metric_raw = "f1"
        secondary_metric_key = "accuracy_mean"
        secondary_metric_label = "Accuracy"

        # Pre-generate CV splits untuk paired comparison
        cv_strategy_name = "StratifiedKFold"
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pre_generated_splits = list(splitter.split(X, y))

    evaluations: list[TreatmentCandidateEvaluation] = []
    candidate_fold_scores: dict[str, list[float]] = {}

    for candidate in candidates:
        # Bangun pipeline scikit-learn untuk kandidat ini
        pipeline, _ = build_pipeline(
            decisions=candidate.decisions,
            profile_features=profile.features,
            model_estimator=benchmark_estimator,
            enable_feature_engineering=False,
        )

        # Jalankan Cross-Validation dengan pre-generated splits bersama (Paired CV)
        cv_result = run_cross_validation(
            pipeline=pipeline,
            X=X,
            y=y,
            model_name=benchmark_name,
            n_splits=n_splits,
            random_seed=seed,
            task_type=task_type,
            cv_splits=pre_generated_splits,
        )

        prim_score = cv_result.summary.get(primary_metric_key, 0.0)
        prim_std = cv_result.summary.get(primary_metric_key.replace("_mean", "_std"), 0.0)
        sec_score = cv_result.summary.get(secondary_metric_key, 0.0)

        # Simpan skor metrik utama mentah per-fold untuk analisis berpasangan auditable
        fold_scores = [
            fr.metrics.get(primary_metric_raw, 0.0)
            for fr in cv_result.fold_results
        ]
        candidate_fold_scores[candidate.candidate_id] = fold_scores

        evaluations.append(
            TreatmentCandidateEvaluation(
                candidate_id=candidate.candidate_id,
                name=candidate.name,
                strategy_type=candidate.strategy_type,
                primary_metric_name=primary_metric_label,
                primary_score=round(prim_score, 4),
                primary_std=round(prim_std, 4),
                secondary_metric_name=secondary_metric_label,
                secondary_score=round(sec_score, 4),
                rank=1,  # akan diperbarui saat pengurutan
                summary_tags=candidate.summary_tags,
                training_time_seconds=round(cv_result.training_time_seconds, 3),
            )
        )

    # Urutkan berdasarkan skor metrik utama (tertinggi ke terendah)
    evaluations.sort(key=lambda e: e.primary_score, reverse=True)
    for rank_idx, e in enumerate(evaluations):
        e.rank = rank_idx + 1

    best_candidate_id = evaluations[0].candidate_id if evaluations else ""

    # Ekstrak skor fold kandidat Recommended dan Baseline
    rec_fold_scores = candidate_fold_scores.get("recommended", [])
    base_fold_scores = candidate_fold_scores.get("baseline", [])
    fold_diffs = [
        r - b for r, b in zip(rec_fold_scores, base_fold_scores)
    ]

    cv_desc = f"{cv_strategy_name} {n_splits}-Fold"

    # Evaluasi paired difference CI dengan penanganan edge cases eksplisit
    if "recommended" not in candidate_fold_scores or "baseline" not in candidate_fold_scores:
        ci_stats = {
            "validation_status": "INVALID_EVALUATION",
            "headline": "Evaluasi Proksi Tidak Valid: Kandidat tidak lengkap",
            "explanation": "Strategi rekomendasi atau baseline tidak ditemukan dalam hasil evaluasi kandidat.",
            "mean_difference": None,
            "relative_difference": None,
            "ci_lower": None,
            "ci_upper": None,
            "sign_agreement": "",
        }
    else:
        ci_stats = compute_paired_difference_ci(
            rec_scores=rec_fold_scores,
            base_scores=base_fold_scores,
            confidence_level=confidence_level,
            cv_desc=cv_desc,
            benchmark_model=benchmark_name,
        )

    # Persistensi Metadata Evaluasi (Requirement 4 & 6)
    evaluation_metadata = {
        "task_type": task_type,
        "ci_method": ci_stats.get("ci_method", "paired_two_tailed_student_t"),
        "degrees_of_freedom": ci_stats.get("degrees_of_freedom"),
        "n_valid_folds": ci_stats.get("n_valid_folds", len(fold_diffs)),
        "estimator": benchmark_name,
        "estimator_parameters": {
            "max_iter": 500 if not is_regression else None,
            "class_weight": None,
            "random_state": seed,
        } if not is_regression else {
            "random_state": seed,
        },
        "cv_strategy": cv_strategy_name,
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": seed,
        "scoring_metric": primary_metric_raw,
        "confidence_level": confidence_level,
    }

    lift_val = ci_stats["mean_difference"] if ci_stats["mean_difference"] is not None else 0.0
    lift_pct_val = ci_stats["relative_difference"] if ci_stats["relative_difference"] is not None else None

    return TreatmentValidationReport(
        task_type=task_type,
        benchmark_model=benchmark_name,
        primary_metric=primary_metric_label,
        validation_status=ci_stats["validation_status"],
        headline=ci_stats["headline"],
        explanation=ci_stats["explanation"],
        lift=lift_val,
        lift_percentage=lift_pct_val,
        best_candidate_id=best_candidate_id,
        candidates=evaluations,
        mean_difference=ci_stats["mean_difference"],
        relative_difference=ci_stats["relative_difference"],
        confidence_level=confidence_level,
        ci_lower=ci_stats["ci_lower"],
        ci_upper=ci_stats["ci_upper"],
        baseline_fold_scores=base_fold_scores,
        recommended_fold_scores=rec_fold_scores,
        fold_differences=fold_diffs,
        sign_agreement=ci_stats["sign_agreement"],
        evaluation_metadata=evaluation_metadata,
    )
