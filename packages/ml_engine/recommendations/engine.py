from __future__ import annotations

"""
=============================================================================
ENGINE REKOMENDASI PREPROCESSING DETERMINISTIK (recommendations/engine.py)
=============================================================================
Fungsi Modul Ini:
Mengambil bukti statistik hasil profiling + konteks dataset → menghasilkan daftar
rekomendasi tindakan pra-pemrosesan data yang terurut berdasarkan skor keyakinan.

Prinsip Utama (ADR-003):
- Deterministik & Berbasis Bukti (Bukan tebakan LLM / acak).
- Tidak pernah mengubah dataset asli (Read-only / Immutable input).
- Setiap rekomendasi membawa: bukti (evidence), alasan (rationale), skor (score 0-100),
  dan opsi tindakan alternatif yang bisa dipilih user.
=============================================================================
"""

import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from scipy import stats

from profiling.profiler import DatasetProfile
from detection.missing_detector import detect_missing
from detection.outlier_detector import detect_outliers
from detection.imbalance_detector import detect_imbalance
from detection.leakage_detector import detect_leakage


# =============================================================================
# 1. STRUKTUR DATA REKOMENDASI (DATACLASS)
# =============================================================================

@dataclass
class Recommendation:
    """
    Menyimpan satu objek rekomendasi tindakan pra-pemrosesan data.
    """
    id: str                   # UUID unik rekomendasi
    category: str             # Kategori: missing_value / outlier / encoding / scaling / class_imbalance
    column: str | None        # Nama kolom yang bersangkutan (None jika level dataset)
    recommended_action: str   # Tindakan utama yang direkomendasikan (misal: median_imputation)
    confidence: float         # Tingkat keyakinan (0.0 s/d 1.0)
    score: float              # Skor rekomendasi (0 s/d 100, bukan probabilitas)
    evidence: list[str]       # Daftar bukti statistik penyebab rekomendasi ini dibuat
    rationale: str            # Penjelasan alasan ilmiah pemilihan tindakan
    alternatives: list[str]   # Daftar opsi tindakan alternatif yang bisa di-override user
    status: str = "proposed"  # Status: proposed / accepted / overridden / rejected

    def to_dict(self) -> dict[str, Any]:
        """Mengubah objek Recommendation menjadi dictionary JSON-serializable."""
        return {
            "id": self.id,
            "category": self.category,
            "column": self.column,
            "recommended_action": self.recommended_action,
            "confidence": round(self.confidence, 3),
            "score": round(self.score, 1),
            "evidence": self.evidence,
            "rationale": self.rationale,
            "alternatives": self.alternatives,
            "status": self.status,
        }


# =============================================================================
# 2. AMBANG BATAS & REKOMENDASI MISSING VALUE
# =============================================================================

# Ambang batas statistik yang dapat dikonfigurasi
HIGH_MISSING_THRESHOLD = 0.40  # Jika > 40% kosong → pertimbangkan drop kolom
HIGH_SKEW_THRESHOLD = 1.0      # Jika |skewness| >= 1.0 → gunakan median alih-alih mean
HIGH_OUTLIER_THRESHOLD = 0.05  # Jika > 5% outlier → gunakan median / robust scaler


def _recommend_missing_numeric(
    column: str,
    missing_rate: float,
    skewness: float | None,
    outlier_rate: float | None,
    sample_size: int,
) -> Recommendation:
    """Menentukan rekomendasi penanganan nilai kosong untuk kolom NUMERIK."""
    evidence = [
        f"missing_rate={round(missing_rate * 100, 1)}%",
        f"dtype=numeric",
    ]
    if skewness is not None:
        evidence.append(f"skewness={round(skewness, 2)}")
    if outlier_rate is not None:
        evidence.append(f"outlier_rate_iqr={round(outlier_rate * 100, 1)}%")
    evidence.append(f"sample_size={sample_size}")

    # Kasus 1: Nilai kosong terlalu banyak (>= 40%) -> Drop Column
    if missing_rate >= HIGH_MISSING_THRESHOLD:
        action = "drop_column"
        confidence = 0.65
        score = 55.0
        rationale = (
            f"{round(missing_rate * 100, 1)}% dari total baris bernilai kosong. "
            "Mengimputasi kolom yang terlalu banyak kosong berisiko menambah noise besar. "
            "Disarankan menghapus kolom kecuali memiliki urgensi domain yang sangat tinggi."
        )
        alternatives = ["median_imputation", "mean_imputation", "knn_imputation"]

    # Kasus 2: Kemiringan distribusi tinggi atau ada outlier -> Median Imputation
    elif (skewness is not None and abs(skewness) >= HIGH_SKEW_THRESHOLD) or (
        outlier_rate is not None and outlier_rate >= HIGH_OUTLIER_THRESHOLD
    ):
        action = "median_imputation"
        confidence = 0.82
        score = 78.0
        rationale = (
            "Distribusi data miring atau memiliki nilai ekstrem (outlier). "
            "Imputasi Median lebih tahan terhadap bias outlier dibandingkan imputasi Mean."
        )
        alternatives = ["mean_imputation", "knn_imputation", "drop_rows"]

    # Kasus 3: Normal / Standar -> Median Imputation (Safe default)
    else:
        action = "median_imputation"
        confidence = 0.70
        score = 65.0
        rationale = (
            "Tingkat nilai kosong moderat dan distribusi data relatif simetris. "
            "Imputasi Median adalah opsi aman bawaan."
        )
        alternatives = ["mean_imputation", "knn_imputation", "drop_rows"]

    return Recommendation(
        id=str(uuid.uuid4()),
        category="missing_value",
        column=column,
        recommended_action=action,
        confidence=confidence,
        score=score,
        evidence=evidence,
        rationale=rationale,
        alternatives=alternatives,
    )


def _recommend_missing_categorical(
    column: str,
    missing_rate: float,
    unique_count: int,
    sample_size: int,
) -> Recommendation:
    """Menentukan rekomendasi penanganan nilai kosong untuk kolom KATEGORIKAL (Teks)."""
    evidence = [
        f"missing_rate={round(missing_rate * 100, 1)}%",
        f"dtype=categorical",
        f"unique_count={unique_count}",
        f"sample_size={sample_size}",
    ]

    if missing_rate >= HIGH_MISSING_THRESHOLD:
        action = "drop_column"
        confidence = 0.60
        score = 52.0
        rationale = (
            f"{round(missing_rate * 100, 1)}% nilai kosong pada kolom kategorikal. "
            "Disarankan menghapus kolom untuk menghindari manipulasi variabel kategorikal buatan."
        )
        alternatives = ["most_frequent_imputation", "constant_imputation"]

    else:
        action = "most_frequent_imputation"
        confidence = 0.75
        score = 70.0
        rationale = (
            "Imputasi dengan Modus (kategori terbanyak) menjaga distribusi kategori asli "
            "tanpa membuat kategori buatan baru."
        )
        alternatives = ["constant_imputation", "drop_rows", "drop_column"]

    return Recommendation(
        id=str(uuid.uuid4()),
        category="missing_value",
        column=column,
        recommended_action=action,
        confidence=confidence,
        score=score,
        evidence=evidence,
        rationale=rationale,
        alternatives=alternatives,
    )


# =============================================================================
# 3. REKOMENDASI OUTLIER
# =============================================================================

def _recommend_outlier(
    column: str,
    outlier_rate_iqr: float,
    outlier_rate_zscore: float,
    skewness: float | None,
) -> Recommendation:
    """Menentukan rekomendasi penanganan Outlier pada kolom numerik."""
    evidence = [
        f"outlier_rate_iqr={round(outlier_rate_iqr * 100, 1)}%",
        f"outlier_rate_robust_z={round(outlier_rate_zscore * 100, 1)}%",
    ]
    if skewness is not None:
        evidence.append(f"skewness={round(skewness, 2)}")

    both_agree = outlier_rate_zscore > 0.01

    if outlier_rate_iqr < 0.02:
        action = "leave_unchanged"
        confidence = 0.80
        score = 72.0
        rationale = (
            "Jumlah outlier sangat sedikit (<2%). Nilai ini mungkin fenomena asli data "
            "sehingga aman biarkan tetap seperti semula (leave unchanged)."
        )
        alternatives = ["winsorization", "iqr_filtering"]
    elif skewness is not None and abs(skewness) > 1.5:
        action = "robust_scaling"
        confidence = 0.72 if both_agree else 0.60
        score = 68.0 if both_agree else 56.0
        rationale = (
            "Fitur sangat miring dan memiliki outlier. RobustScaler menggunakan Median & IQR "
            "sehingga skala data tidak terpengaruh nilai ekstrem."
        )
        alternatives = ["winsorization", "iqr_filtering", "leave_unchanged"]
    else:
        action = "winsorization"
        confidence = 0.68 if both_agree else 0.55
        score = 63.0 if both_agree else 50.0
        rationale = (
            "Winsorization membatasi nilai ekstrem pada batas IQR fence "
            "tanpa perlu menghapus baris data dari dataset."
        )
        alternatives = ["leave_unchanged", "iqr_filtering", "robust_scaling"]

    return Recommendation(
        id=str(uuid.uuid4()),
        category="outlier",
        column=column,
        recommended_action=action,
        confidence=confidence,
        score=score,
        evidence=evidence,
        rationale=rationale,
        alternatives=alternatives,
    )


# =============================================================================
# 4. REKOMENDASI ENCODING (UBAH TEKS MENJADI ANGKA)
# =============================================================================

def _recommend_encoding(
    column: str,
    unique_count: int,
    sample_size: int,
) -> Recommendation:
    """Menentukan tipe encoding terbaik (One-Hot, Ordinal, atau Frequency)."""
    evidence = [
        f"unique_count={unique_count}",
        f"sample_size={sample_size}",
    ]

    HIGH_CARD = 50

    # Kardinalitas Tinggi (> 50 nilai unik) -> Frequency Encoding
    if unique_count > HIGH_CARD:
        action = "frequency_encoding"
        confidence = 0.60
        score = 55.0
        rationale = (
            f"Kardinalitas tinggi ({unique_count} kategori unik). One-Hot encoding akan membuat "
            f"terlalu banyak kolom baru (curse of dimensionality). Frequency encoding lebih aman."
        )
        evidence.append("warning=high_cardinality")
        alternatives = ["ordinal_encoding", "drop_column"]

    # Biner (2 nilai unik) -> Ordinal Encoding (0/1)
    elif unique_count <= 2:
        action = "ordinal_encoding"
        confidence = 0.88
        score = 85.0
        rationale = (
            "Fitur kategorikal biner (2 nilai unik). Ordinal encoding (0/1) sangat ringkas dan optimal."
        )
        alternatives = ["one_hot_encoding"]

    # Kardinalitas Rendah/Sedang -> One-Hot Encoding
    else:
        action = "one_hot_encoding"
        confidence = 0.82
        score = 78.0
        rationale = (
            f"Kardinalitas sedang ({unique_count} kategori). One-Hot encoding adalah standar "
            "untuk mencegah model memprediksikan tingkatan implisit antar kategori."
        )
        alternatives = ["ordinal_encoding", "frequency_encoding"]

    return Recommendation(
        id=str(uuid.uuid4()),
        category="encoding",
        column=column,
        recommended_action=action,
        confidence=confidence,
        score=score,
        evidence=evidence,
        rationale=rationale,
        alternatives=alternatives,
    )


# =============================================================================
# 5. REKOMENDASI SCALING (PENYESUAIAN SKALA ANGKA)
# =============================================================================

def _recommend_scaling(
    column: str,
    skewness: float | None,
    outlier_rate: float | None,
    feature_range: float | None,
) -> Recommendation:
    """Menentukan tipe Scaler terbaik (StandardScaler, RobustScaler, MinMaxScaler)."""
    evidence = []
    if skewness is not None:
        evidence.append(f"skewness={round(skewness, 2)}")
    if outlier_rate is not None:
        evidence.append(f"outlier_rate_iqr={round(outlier_rate * 100, 1)}%")
    if feature_range is not None:
        evidence.append(f"range={round(feature_range, 2)}")

    if outlier_rate and outlier_rate > HIGH_OUTLIER_THRESHOLD:
        action = "robust_scaling"
        confidence = 0.80
        score = 76.0
        rationale = (
            "Fitur memiliki outlier signifikan. RobustScaler menggunakan Median & IQR "
            "sehingga skala tidak rusak oleh nilai ekstrem."
        )
        alternatives = ["standard_scaling", "no_scaling"]

    elif skewness is not None and abs(skewness) < 0.5:
        action = "standard_scaling"
        confidence = 0.78
        score = 74.0
        rationale = (
            "Distribusi data relatif simetris tanpa outlier. "
            "StandardScaler (mean=0, variance=1) adalah opsi terbaik."
        )
        alternatives = ["min_max_scaling", "robust_scaling", "no_scaling"]

    else:
        action = "standard_scaling"
        confidence = 0.70
        score = 66.0
        rationale = (
            "StandardScaler adalah opsi standar yang baik untuk algoritma berbahan dasar jarak/gradien."
        )
        alternatives = ["robust_scaling", "min_max_scaling", "no_scaling"]

    return Recommendation(
        id=str(uuid.uuid4()),
        category="scaling",
        column=column,
        recommended_action=action,
        confidence=confidence,
        score=score,
        evidence=evidence,
        rationale=rationale,
        alternatives=alternatives,
    )


# =============================================================================
# 6. REKOMENDASI LEAKAGE (KEBOCORAN DATA)
# =============================================================================

def _recommend_leakage(
    leakage_type: str,
    column: str,
    severity: str,
    correlation_value: float | None,
    correlation_method: str | None,
    unique_ratio: float | None,
    evidence: list[str],
    recommendation_text: str,
) -> Recommendation:
    """
    Membuat Recommendation dari hasil LeakageReport.

    Score diberikan berdasarkan severity:
    - critical: 95 (tertinggi, harus ditangani sebelum training)
    - high:     88
    - medium:   75
    """
    severity_score = {"critical": 95.0, "high": 88.0, "medium": 75.0}
    severity_conf  = {"critical": 0.97, "high": 0.90, "medium": 0.80}

    score = severity_score.get(severity, 75.0)
    confidence = severity_conf.get(severity, 0.80)

    if leakage_type == "identifier":
        recommended_action = "drop_column"
        rationale = (
            f"Kolom '{column}' terdeteksi sebagai identifier (rasio unik = "
            f"{round((unique_ratio or 0) * 100, 1)}%). "
            "Identifier tidak membawa informasi prediktif dan menyebabkan "
            "overfitting sempurna (model menghafal ID, bukan belajar pola)."
        )
        alternatives = ["verify_manually"]
    else:  # target_proxy
        recommended_action = "drop_column"
        corr_str = f"{correlation_method}={round(correlation_value or 0, 4)}"
        rationale = (
            f"Kolom '{column}' memiliki korelasi sangat tinggi dengan target ({corr_str}). "
            "Kemungkinan besar merupakan proxy bocoran dari target yang dikumpulkan "
            "setelah label diketahui — menyebabkan model tidak bisa generalisasi ke data baru."
        )
        alternatives = ["verify_manually", "keep_if_valid_feature"]

    return Recommendation(
        id=str(uuid.uuid4()),
        category="data_leakage",
        column=column,
        recommended_action=recommended_action,
        confidence=confidence,
        score=score,
        evidence=evidence,
        rationale=rationale,
        alternatives=alternatives,
    )


# =============================================================================
# 7. REKOMENDASI CLASS IMBALANCE (KETIDAKSEIMBANGAN KELAS TARGET)
# =============================================================================

def _recommend_imbalance(
    minority_class: str,
    minority_prop: float,
    imbalance_ratio: float,
) -> Recommendation:
    """Rekomendasi penanganan ketidakseimbangan kelas pada kolom target."""
    evidence = [
        f"minority_class={minority_class}",
        f"minority_proportion={round(minority_prop * 100, 1)}%",
        f"imbalance_ratio={round(imbalance_ratio, 1)}:1",
    ]
    return Recommendation(
        id=str(uuid.uuid4()),
        category="class_imbalance",
        column=None,
        recommended_action="use_class_weight",
        confidence=0.75,
        score=70.0,
        evidence=evidence,
        rationale=(
            f"Dataset mengalami ketidakseimbangan kelas (rasio {round(imbalance_ratio, 1)}:1). "
            "Mengaktifkan class_weight='balanced' pada classifier adalah solusi pertama yang efisien."
        ),
        alternatives=["stratified_cv_only", "oversample_smote", "undersample"],
    )


def _recommend_target_skewness(
    target_col: str,
    skewness: float,
    min_val: float,
    max_val: float,
) -> Recommendation:
    """Rekomendasi penanganan kemiringan distribusi (skewness) pada target regresi."""
    is_non_negative = min_val >= 0
    recommended_action = "log1p_transform" if is_non_negative else "yeo_johnson_transform"
    alternatives = ["yeo_johnson_transform", "keep_original"] if is_non_negative else ["keep_original"]

    evidence = [
        f"target_skewness={round(skewness, 2)}",
        f"target_min={round(min_val, 2)}",
        f"target_max={round(max_val, 2)}",
    ]

    rationale = (
        f"Target regresi '{target_col}' mengalami kemiringan distribusi yang signifikan (skewness = {round(skewness, 2)}). "
        f"Menerapkan transformasi '{recommended_action}' dapat menstabilkan variansi residual dan mempermudah konvergensi model regresi."
    )

    return Recommendation(
        id=str(uuid.uuid4()),
        category="target_transformation",
        column=target_col,
        recommended_action=recommended_action,
        confidence=0.85,
        score=75.0,
        evidence=evidence,
        rationale=rationale,
        alternatives=alternatives,
    )


def _recommend_target_outliers(
    target_col: str,
    outlier_rate: float,
    outlier_count: int,
) -> Recommendation:
    """Rekomendasi mitigasi outlier ekstrim pada target regresi."""
    evidence = [
        f"target_outlier_rate={round(outlier_rate * 100, 1)}%",
        f"outlier_count={outlier_count}",
    ]
    rationale = (
        f"Ditemukan {round(outlier_rate * 100, 1)}% data ekstrim (outlier) pada target '{target_col}'. "
        "Disarankan memprioritaskan model regresi yang tangguh (robust) terhadap outlier seperti Random Forest, Gradient Boosting, atau Ridge."
    )
    return Recommendation(
        id=str(uuid.uuid4()),
        category="target_outliers",
        column=target_col,
        recommended_action="use_robust_models",
        confidence=0.75,
        score=65.0,
        evidence=evidence,
        rationale=rationale,
        alternatives=["clip_target_iqr", "keep_original"],
    )


# =============================================================================
# 7. FUNGSI UTAMA GENERASI REKOMENDASI
# =============================================================================

def generate_recommendations(
    df: pd.DataFrame,
    profile: DatasetProfile,
    task_type: str = "classification",
) -> list[Recommendation]:
    """
    Fungsi Utama: Menghasilkan seluruh daftar rekomendasi preprocessing untuk dataset
    secara task-aware (membedakan klasifikasi vs regresi).

    Args:
        df: Dataframe dataset asli (read-only).
        profile: Objek DatasetProfile hasil profiling.
        task_type: Jenis task ("classification" atau "regression"). Defaults to "classification".

    Returns:
        Daftar objek Recommendation yang diurutkan dari skor tertinggi ke terendah.
    """
    recommendations: list[Recommendation] = []
    feature_map = {f.name: f for f in profile.features}
    feature_dtypes = {f.name: f.dtype for f in profile.features}

    # 0. Deteksi Kebocoran Data (Data Leakage) — Prioritas Tertinggi
    if profile.target:
        leakage_reports = detect_leakage(
            df=df,
            target_column=profile.target,
            feature_dtypes=feature_dtypes,
        )
        for lr in leakage_reports:
            rec = _recommend_leakage(
                leakage_type=lr.leakage_type,
                column=lr.column,
                severity=lr.severity,
                correlation_value=lr.correlation_value,
                correlation_method=lr.correlation_method,
                unique_ratio=lr.unique_ratio,
                evidence=lr.evidence,
                recommendation_text=lr.recommendation,
            )
            recommendations.append(rec)

    # 1. Analisis Nilai Kosong (Missing Values)
    missing_reports = detect_missing(df, profile.features)
    for report in missing_reports:
        f = feature_map[report.column]
        if report.dtype == "numeric":
            rec = _recommend_missing_numeric(
                column=report.column,
                missing_rate=report.missing_rate,
                skewness=report.skewness,
                outlier_rate=report.outlier_rate,
                sample_size=profile.rows,
            )
        elif report.dtype in ("categorical", "boolean"):
            rec = _recommend_missing_categorical(
                column=report.column,
                missing_rate=report.missing_rate,
                unique_count=f.unique_count,
                sample_size=profile.rows,
            )
        else:
            continue
        recommendations.append(rec)

    # 2. Analisis Outlier
    outlier_reports = detect_outliers(df, profile.features)
    for report in outlier_reports:
        rec = _recommend_outlier(
            column=report.column,
            outlier_rate_iqr=report.outlier_rate_iqr,
            outlier_rate_zscore=report.outlier_rate_zscore,
            skewness=report.skewness,
        )
        recommendations.append(rec)

    # 3. Analisis Encoding Kategori
    for f in profile.features:
        if f.dtype in ("categorical", "boolean") and not f.constant:
            rec = _recommend_encoding(
                column=f.name,
                unique_count=f.unique_count,
                sample_size=profile.rows,
            )
            recommendations.append(rec)

    # 4. Analisis Scaling Angka
    for f in profile.features:
        if f.dtype == "numeric" and not f.constant:
            feature_range = (f.max - f.min) if (f.max is not None and f.min is not None) else None
            rec = _recommend_scaling(
                column=f.name,
                skewness=f.skewness,
                outlier_rate=f.outlier_rate_iqr,
                feature_range=feature_range,
            )
            recommendations.append(rec)

    # 5. Analisis Target Spesifik Task (Task-Aware Processing)
    if task_type == "regression":
        if profile.target and profile.target in df.columns:
            target_series = pd.to_numeric(df[profile.target], errors="coerce").dropna()
            if len(target_series) >= 5:
                # 5.1 Analisis Kemiringan Distribusi (Skewness) Target
                try:
                    skew_val = float(stats.skew(target_series))
                    if abs(skew_val) >= 1.2:
                        rec = _recommend_target_skewness(
                            target_col=profile.target,
                            skewness=skew_val,
                            min_val=float(target_series.min()),
                            max_val=float(target_series.max()),
                        )
                        recommendations.append(rec)
                except Exception:
                    pass

                # 5.2 Analisis Outlier Ekstrim pada Target
                try:
                    q25 = float(target_series.quantile(0.25))
                    q75 = float(target_series.quantile(0.75))
                    iqr = q75 - q25
                    if iqr > 0:
                        outliers = ((target_series < q25 - 1.5 * iqr) | (target_series > q75 + 1.5 * iqr)).sum()
                        outlier_rate = float(outliers / len(target_series))
                        if outlier_rate >= 0.03:
                            rec = _recommend_target_outliers(
                                target_col=profile.target,
                                outlier_rate=outlier_rate,
                                outlier_count=int(outliers),
                            )
                            recommendations.append(rec)
                except Exception:
                    pass
    else:
        # Task Classification: Analisis Ketidakseimbangan Kelas Target
        imbalance_report = detect_imbalance(profile.class_distribution)
        if imbalance_report and imbalance_report.is_imbalanced:
            rec = _recommend_imbalance(
                minority_class=imbalance_report.minority_class,
                minority_prop=imbalance_report.minority_proportion,
                imbalance_ratio=imbalance_report.imbalance_ratio,
            )
            recommendations.append(rec)

    # Urutkan berdasarkan skor terbesar ke terkecil
    recommendations.sort(key=lambda r: r.score, reverse=True)
    return recommendations

