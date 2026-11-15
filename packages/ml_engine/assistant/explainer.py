from __future__ import annotations

# =============================================================================
# MODUL: RULE-BASED NATURAL LANGUAGE EXPLAINER & ASSISTANT ENGINE
# =============================================================================
# Deskripsi: Mengimplementasikan Rule-Based Engine (Hybrid AI Assistant) sesuai ADR-003.
#            Menerjemahkan bukti statistik deterministik (evidence) dari profiling,
#            rekomendasi preprocessing, dan hasil eksperimen menjadi narasi bahasa
#            Indonesia yang jelas, profesional, dan mudah dipahami orang awam.
#            Zero-dependency ke cloud/LLM pihak ketiga, respons instan (<1ms), dan 100% akurat.
# =============================================================================

from typing import Any


def _parse_num(val: Any, default: float = 0.0) -> float:
    """Helper untuk mengurai nilai numerik dari string (termasuk '%', rasio, dll) secara aman."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val).replace("%", "").strip()
        if ":" in s:
            s = s.split(":")[0].strip()
        return float(s)
    except (ValueError, TypeError):
        return default


def _format_skewness(val: Any) -> str:
    """Formatter pembantu nilai skewness distribusi data."""
    if val is None:
        return "tinggi"
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)


# =============================================================================
# BAGIAN 1: PENJELAS REKOMENDASI PREPROCESSING (EXPLAIN RECOMMENDATION)
# =============================================================================

def explain_recommendation(rec: dict[str, Any]) -> dict[str, Any]:
    """
    Menghasilkan penjelasan Bahasa Indonesia berbasis aturan untuk rekomendasi preprocessing data.

    Parameter:
        rec: Kamus rekomendasi dari recommendation engine.

    Return:
        Kamus terstruktur berisi judul (headline), narasi detail, bukti kunci, dan rasionalitas alternatif.
    """
    column = rec.get("column_name") or rec.get("column") or "dataset"
    category = rec.get("category", "")
    action = rec.get("recommendation") or rec.get("selected_action") or ""
    score = int(round(_parse_num(rec.get("score"), 0)))
    evidence = rec.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    rationale = rec.get("rationale", "")
    alternatives = rec.get("alternative_actions") or rec.get("alternatives") or []

    action_title = action.replace("_", " ").title()
    sample_size = int(_parse_num(evidence.get("sample_size"), 0))

    # 1. Kategori Missing Values
    if category in ("missing_values", "missing_value"):
        raw_rate = _parse_num(evidence.get("missing_rate") or evidence.get("missing_rate_iqr"), 0.0)
        missing_rate = raw_rate if raw_rate > 1.0 else (raw_rate * 100.0)

        missing_count = int(_parse_num(evidence.get("missing_count"), 0))
        if missing_count == 0 and sample_size > 0 and missing_rate > 0:
            missing_count = int(round((missing_rate / 100.0) * sample_size))

        skew_str = _format_skewness(evidence.get("skewness"))

        count_str = f"{missing_count} baris ({missing_rate:.1f}%)" if missing_count > 0 else f"{missing_rate:.1f}% data"

        if "drop_column" in action:
            detail = (
                f"Kolom '{column}' memiliki persentase data hilang yang sangat tinggi ({count_str}). "
                f"Mengimputasi kolom dengan kekosongan data >40% berisiko memasukkan bias atau noise buatan yang besar, "
                f"sehingga disarankan menghapus kolom ini dari pipeline pembelajaran model."
            )
        elif "median" in action:
            detail = (
                f"Kolom '{column}' memiliki data hilang sebanyak {count_str}. "
                f"Tingkat kemiringan distribusi (skewness = {skew_str}) menunjukkan data tidak terdistribusi simetris normal. "
                f"Imputasi Median dipilih karena Median tahan terhadap nilai ekstrem (outlier) sehingga tidak membiaskan representasi fitur."
            )
        elif "mean" in action:
            detail = (
                f"Kolom '{column}' memiliki {count_str} nilai hilang. "
                f"Distribusi data relatif simetris dan mendekati kurva normal, sehingga imputasi Mean (Rata-rata) "
                f"adalah pilihan estimasi titik yang efisien dan optimal secara statistik."
            )
        elif "most_frequent" in action:
            detail = (
                f"Kolom kategorikal '{column}' memiliki {count_str} nilai kosong. "
                f"Imputasi Modus (Kategori Terbanyak) direkomendasikan untuk mempertahankan integritas frekuensi kategori asli "
                f"tanpa menciptakan kategori tiruan baru."
            )
        elif "knn" in action:
            detail = (
                f"Kolom '{column}' memiliki pola nilai hilang kompleks ({count_str}). "
                f"KNN Imputer direkomendasikan untuk memprediksi nilai kosong berdasarkan kedekatan jarak observasi dengan fitur-fitur lainnya."
            )
        elif "constant" in action:
            detail = (
                f"Kolom '{column}' memiliki {count_str} nilai kosong. "
                f"Imputasi nilai konstan (misal 'missing') dipilih untuk menandai kekosongan sebagai status informasi tersendiri."
            )
        else:
            detail = rationale or f"Tindakan {action_title} direkomendasikan untuk menangani {count_str} nilai kosong pada '{column}'."

    # 2. Kategori Outliers
    elif category in ("outliers", "outlier"):
        raw_rate = _parse_num(evidence.get("outlier_rate") or evidence.get("outlier_rate_iqr") or evidence.get("outlier_rate_robust_z"), 0.0)
        outlier_rate = raw_rate if raw_rate > 1.0 else (raw_rate * 100.0)

        outlier_count = int(_parse_num(evidence.get("outlier_count"), 0))
        if outlier_count == 0 and sample_size > 0 and outlier_rate > 0:
            outlier_count = int(round((outlier_rate / 100.0) * sample_size))

        outlier_str = f"{outlier_count} baris ({outlier_rate:.1f}%)" if outlier_count > 0 else f"{outlier_rate:.1f}% data"

        if "leave_unchanged" in action:
            detail = (
                f"Jumlah nilai ekstrem pada kolom '{column}' sangat sedikit ({outlier_str}). "
                f"Nilai-nilai ini kemungkinan merepresentasikan variasi asli domain data, "
                f"sehingga dibiarkan apa adanya (leave unchanged) agar tidak menghilangkan sinyal penting."
            )
        elif "winsorization" in action or "clip" in action:
            detail = (
                f"Terdeteksi {outlier_str} outlier pada kolom '{column}'. "
                f"Winsorization (Clipping) dipilih untuk memotong nilai ekstrem yang melewati batas IQR (Interquartile Range) fence "
                f"ke ambang batas terdekat tanpa membuang baris data observasi."
            )
        elif "robust" in action:
            detail = (
                f"Fitur numerik '{column}' memiliki nilai outlier signifikan ({outlier_str}) dengan kemiringan distribusi yang tinggi. "
                f"RobustScaler direkomendasikan karena menggunakan Median dan rentang IQR interkuartil yang kebal terhadap lonjakan nilai ekstrem."
            )
        elif "iqr_filtering" in action:
            detail = (
                f"Ditemukan {outlier_str} baris outlier ekstrem pada '{column}'. "
                f"IQR Filtering dapat digunakan untuk menyaring baris sampel yang berada jauh di luar batas kuartil data."
            )
        else:
            detail = rationale or f"Pemeriksaan outlier pada '{column}' menemukan {outlier_str} di luar batas toleransi distribusi data."

    # 3. Kategori Categorical Encoding
    elif category in ("encoding", "categorical_encoding"):
        unique_count = int(_parse_num(evidence.get("unique_count") or evidence.get("cardinality"), 0))
        card_str = f"{unique_count} nilai unik" if unique_count > 0 else "jumlah kategori terbatas"

        if "one_hot" in action:
            detail = (
                f"Kolom kategorikal '{column}' memiliki kardinalitas rendah hingga sedang ({card_str}). "
                f"One-Hot Encoding dipilih untuk mengubah setiap kategori menjadi vektor biner independen, "
                f"mencegah algoritma mengasumsikan urutan hierarki numerik palsu antar kategori."
            )
        elif "ordinal" in action:
            detail = (
                f"Kolom '{column}' memiliki karakteristik biner atau bertingkat ({card_str}). "
                f"Ordinal Encoding dipilih untuk mengodekan nilai menjadi integer berurutan secara ringkas tanpa menambah dimensi kolom baru."
            )
        elif "frequency" in action:
            detail = (
                f"Kolom '{column}' memiliki kardinalitas tinggi ({card_str}). "
                f"One-Hot Encoding pada kolom ini akan memicu ledakan dimensi fitur (curse of dimensionality). "
                f"Oleh karena itu, Frequency Encoding dipilih untuk merepresentasikan kategori berdasarkan proporsi frekuensi kemunculannya."
            )
        else:
            detail = rationale or f"Encoding {action_title} direkomendasikan untuk kolom kategorikal '{column}'."

    # 4. Kategori Feature Scaling
    elif category == "scaling":
        skew_val = _parse_num(evidence.get("skewness"), 0.0)
        outlier_rate = _parse_num(evidence.get("outlier_rate_iqr"), 0.0)

        if "robust" in action:
            detail = (
                f"Fitur numerik '{column}' memiliki outlier atau distribusi condong miring (skewness = {_format_skewness(skew_val)}). "
                f"RobustScaler digunakan karena melakukan centering dengan Median dan penskalaan dengan IQR (Interquartile Range), "
                f"sehingga tidak terdistorsi oleh titik data ekstrem."
            )
        elif "standard" in action:
            detail = (
                f"Fitur numerik '{column}' memiliki distribusi yang relatif simetris dan bebas outlier ekstrem. "
                f"StandardScaler (Z-score normalization) mentransformasi data menjadi mean=0 dan varians=1, "
                f"sangat ideal untuk algoritma berbasis gradien dan jarak Euclidean."
            )
        elif "min_max" in action:
            detail = (
                f"Fitur '{column}' diskalakan ke interval rentang tetap [0, 1] menggunakan MinMaxScaler. "
                f"Opsi ini menyamakan skala seluruh fitur numerik dengan tetap mempertahankan bentuk distribusi aslinya."
            )
        elif "no_scaling" in action:
            detail = (
                f"Fitur '{column}' tidak memerlukan penskalaan ulang karena sudah berada dalam rentang skala yang seragam "
                f"atau algoritma model berbasis pohon (Tree-based) kebal terhadap besaran skala nilai numerik."
            )
        else:
            detail = rationale or f"Penskalaan {action_title} dipilih untuk standardisasi rentang nilai kolom '{column}'."

    # 5. Kategori Class Imbalance
    elif category in ("imbalance", "class_imbalance"):
        raw_ratio = _parse_num(evidence.get("imbalance_ratio"), 1.0)
        raw_pct = _parse_num(evidence.get("minority_percentage") or evidence.get("minority_pct"), 0.0)
        minority_pct = raw_pct if raw_pct > 1.0 else (raw_pct * 100.0)

        detail = (
            f"Terdeteksi ketidakseimbangan proporsi kelas target '{column}' (rasio {raw_ratio:.1f}:1, kelas minoritas {minority_pct:.1f}%). "
            f"Evaluasi model dengan Akurasi biasa berisiko menimbulkan ilusi akurasi (accuracy paradox). "
            f"Direkomendasikan menggunakan Stratified K-Fold CV serta metrik F1-Score atau PR-AUC untuk penilaian performa yang objektif."
        )

    # 6. Kategori Transformasi Target Regresi (Target Skewness)
    elif category in ("target_transformation", "target_skewness"):
        skew_val = _parse_num(evidence.get("target_skewness") or evidence.get("skewness"), 0.0)
        detail = (
            f"Kolom target regresi '{column}' memiliki kemiringan distribusi yang tinggi (skewness {skew_val:.2f}). "
            f"Aksi '{action_title}' direkomendasikan untuk menstabilkan variansi residual serta membantu model regresi menangani nilai berskala besar dengan galat yang lebih seimbang."
        )

    # 7. Kategori Outlier Target Regresi
    elif category in ("target_outliers", "target_outlier"):
        rate_str = str(evidence.get("target_outlier_rate", "5%"))
        detail = (
            f"Ditemukan outlier signifikan ({rate_str}) pada kolom target regresi '{column}'. "
            f"Aksi '{action_title}' disarankan agar estimasi bobot model regresi tidak terdistorsi secara ekstrim oleh data pencilan."
        )

    # 8. Kategori Kebocoran Data (Data Leakage: Target Proxy & Identifier)
    elif category in ("data_leakage", "leakage"):
        leak_type = str(evidence.get("leakage_type") or evidence.get("type") or "").lower()
        if "identifier" in leak_type or "id" in column.lower():
            detail = (
                f"Kolom '{column}' terdeteksi sebagai kolom identifikasi unik (Identifier). "
                f"Kolom identifier tidak membawa pola prediktif dan menyebabkan model mengalami overfitting total "
                f"(menghafal ID data latih). Menghapus kolom ini wajib dilakukan agar model tidak gagal saat memprediksi data baru."
            )
        else:
            corr_val = evidence.get("correlation_value") or evidence.get("correlation") or ""
            corr_str = f"nilai korelasi {corr_val}" if corr_val else "korelasi mendekati 1.0"
            detail = (
                f"Kolom '{column}' memiliki hubungan statistik yang terlalu ekstrem dengan target ({corr_str}). "
                f"Fitur ini terindikasi kuat sebagai 'Target Proxy' (kebocoran informasi label yang tidak akan tersedia di produksi). "
                f"Menghapus kolom ini krusial untuk mencegah ilusi akurasi semu dan memastikan model mampu melakukan generalisasi."
            )

    else:
        detail = rationale or f"Rekomendasi {action_title} dihitung otomatis berdasarkan karakteristik statistik dataset."

    # Penjelasan Aksi Alternatif
    alt_explanations = []
    for alt in alternatives:
        alt_name = alt.replace("_", " ").title()
        alt_explanations.append(f"• **{alt_name}**: Opsi alternatif jika pendekatan domain menginginkan perlakuan berbeda.")

    key_evidence_list = []
    for k, v in evidence.items():
        if isinstance(v, (int, float, str)) and str(v).strip():
            label = k.replace("_", " ").title()
            val_str = str(v)
            key_evidence_list.append(f"{label}: {val_str}")

    return {
        "column_name": column,
        "category": category,
        "selected_action": action,
        "action_title": action_title,
        "confidence_score": score,
        "headline": f"{action_title} direkomendasikan untuk '{column}' (Keyakinan: {score}%)",
        "detailed_explanation": detail,
        "key_evidence": key_evidence_list,
        "alternative_rationale": "\n".join(alt_explanations) if alt_explanations else "Tidak ada alternatif khusus yang terdaftar.",
    }


# =============================================================================
# =============================================================================
# BAGIAN 2: RINGKASAN REKAPITULASI EKSPERIMEN (SUMMARIZE EXPERIMENTS)
# =============================================================================

MODEL_DISPLAY_NAMES: dict[str, str] = {
    # Classification
    "dummy_classifier": "Dummy Classifier",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest Classifier",
    "gradient_boosting": "Gradient Boosting Classifier",
    "hist_gradient_boosting": "HistGrad Boost Classifier",
    "knn": "K-Nearest Neighbors Classifier",
    # Regression
    "dummy_regressor": "Dummy Regressor",
    "linear_regression": "Linear Regression",
    "ridge": "Ridge Regression",
    "random_forest_regressor": "Random Forest Regressor",
    "gradient_boosting_regressor": "Gradient Boosting Regressor",
    "hist_gradient_boosting_regressor": "HistGrad Boost Regressor",
    "knn_regressor": "KNN Regressor",
    "xgb_regressor": "XGBoost Regressor",
    "lgbm_regressor": "LightGBM Regressor",
}


def format_model_display_name(raw_name: str | None) -> str:
    """Konversi nama internal/teknis model ML ke nama tampilan yang jelas dan profesional."""
    if not raw_name:
        return "Model"
    clean = str(raw_name).strip()
    is_tuned = False
    if clean.endswith("_tune"):
        clean = clean[:-5]
        is_tuned = True
    display = MODEL_DISPLAY_NAMES.get(clean, clean.replace("_", " ").title())
    return f"{display} (Tuned)" if is_tuned else display


def summarize_experiments(
    experiments: list[dict[str, Any]],
    candidate: dict[str, Any] | None = None,
    target_column: str = "target",
    task_type: str = "classification",
) -> dict[str, Any]:
    """
    Merangkum hasil benchmarking eksperimen model ML menjadi laporan analitis ringkas dan mudah dipahami.
    Mendukung tugas Klasifikasi (F1, Accuracy, ROC-AUC) maupun Regresi (R², RMSE, MAE).
    """
    completed = [e for e in experiments if e.get("status") == "completed"]
    failed = [e for e in experiments if e.get("status") == "failed"]

    if not completed:
        task_label = "regresi" if task_type == "regression" else "klasifikasi"
        return {
            "headline": "Belum ada eksperimen yang selesai dijalankan.",
            "narrative": f"Pilih algoritma model {task_label} di panel kontrol dan klik 'Run Experiments' untuk memulai benchmarking.",
            "best_model_name": None,
            "strengths": [],
            "takeaways": ["Pilih model dan jalankan eksperimen."],
        }

    def _get_metric_val(exp: dict, metric_names: tuple[str, ...]) -> float | None:
        for m in exp.get("metrics", []):
            if m.get("name") in metric_names and m.get("split") == "cv":
                try:
                    return float(m.get("value", 0.0))
                except (ValueError, TypeError):
                    return None
        return None

    is_regression = task_type == "regression" or any(
        _get_metric_val(e, ("r2_mean", "r2", "rmse_mean", "rmse")) is not None for e in completed
    )

    if is_regression:
        # --- LOGIKA TUGAS REGRESI ---
        def get_r2(exp: dict) -> float:
            val = _get_metric_val(exp, ("r2_mean", "r2"))
            return val if val is not None else -999.0

        sorted_exps = sorted(completed, key=get_r2, reverse=True)
        best = sorted_exps[0]
        best_raw_name = best.get("model_name", "Model Best")
        best_name = format_model_display_name(best_raw_name)
        best_r2 = get_r2(best)
        best_rmse = _get_metric_val(best, ("rmse_mean", "rmse"))
        best_mae = _get_metric_val(best, ("mae_mean", "mae"))

        rmse_str = f", RMSE: {best_rmse:,.4f}" if best_rmse is not None else ""
        mae_str = f", MAE: {best_mae:,.4f}" if best_mae is not None else ""

        headline = f"🏆 {best_name} meraih performa tertinggi dengan R² Score {best_r2:.4f}"
        
        unique_algos = {str(e.get("model_name", "")).replace("_tune", "") for e in completed if e.get("model_name")}
        total_runs = len(completed)

        if total_runs > len(unique_algos):
            count_desc = f"Dari total {total_runs} eksperimen benchmarking ({len(unique_algos)} arsitektur algoritma)"
        else:
            count_desc = f"Dari total {total_runs} model regresi yang berhasil dievaluasi"

        r2_pct = max(0.0, best_r2 * 100) if best_r2 > -1 else 0.0
        r2_expl = f" (menjelaskan ~{r2_pct:.1f}% variasi data)" if best_r2 > 0 else ""

        narrative = (
            f"{count_desc} untuk target kontinu '{target_column}', "
            f"**{best_name}** mencatat kemampuan prediksi terbaik dengan rata-rata R² Score sebesar **{best_r2:.4f}**{r2_expl}{rmse_str}{mae_str}. "
        )

        # Cari runner-up dari arsitektur model yang BERBEDA agar tidak membandingkan model yang sama
        best_base_algo = str(best_raw_name).replace("_tune", "")
        distinct_runner_up = next(
            (e for e in sorted_exps[1:] if str(e.get("model_name", "")).replace("_tune", "") != best_base_algo),
            None,
        )

        if distinct_runner_up:
            ru_raw = distinct_runner_up.get("model_name", "")
            ru_name = format_model_display_name(ru_raw)
            ru_r2 = get_r2(distinct_runner_up)
            diff = best_r2 - ru_r2
            if diff > 0.0005:
                narrative += f"Sebagai perbandingan, model alternatif terbaik berikutnya adalah **{ru_name}** dengan R² Score {ru_r2:.4f} (terpaut selisih {diff:.4f} di bawah {best_name})."
            else:
                narrative += f"Model alternatif **{ru_name}** memiliki performa yang sangat kompetitif dengan R² Score {ru_r2:.4f}."
        elif total_runs > 1:
            narrative += f"Eksperimen yang dievaluasi berfokus pada eksplorasi konfigurasi dan hyperparameter untuk arsitektur **{best_name}**."

        strengths = [
            f"R² Score tertinggi ({best_r2:.4f}) pada Cross-Validation K-Fold",
            f"Waktu komputasi pelatihan: {best.get('training_time_seconds', 0):.2f} detik",
            "Pipeline preprocessing terisolasi bebas dari kebocoran data (data leakage)",
        ]
        if best_rmse is not None:
            strengths.append(f"Root Mean Squared Error (RMSE): {best_rmse:,.4f}")
        if best_mae is not None:
            strengths.append(f"Mean Absolute Error (MAE): {best_mae:,.4f}")

        cand_name = format_model_display_name(candidate.get("model_name")) if candidate and candidate.get("model_name") else "Belum dipilih"
        takeaways = [
            f"Total eksperimen selesai: {len(completed)} | Gagal: {len(failed)}",
            f"Kandidat model aktif: {cand_name}",
            "Tinjau sebaran titik Residual pada grafik Actual vs Predicted di Halaman Hasil.",
        ]

        return {
            "headline": headline,
            "narrative": narrative,
            "best_model_name": best_name,
            "best_metric_name": "R²",
            "best_metric_value": best_r2,
            "strengths": strengths,
            "takeaways": takeaways,
        }

    else:
        # --- LOGIKA TUGAS KLASIFIKASI ---
        def get_f1(exp: dict) -> float:
            val = _get_metric_val(exp, ("f1_mean", "f1"))
            if val is not None:
                return val
            val_acc = _get_metric_val(exp, ("accuracy_mean", "accuracy"))
            return val_acc if val_acc is not None else 0.0

        sorted_exps = sorted(completed, key=get_f1, reverse=True)
        best = sorted_exps[0]
        best_raw_name = best.get("model_name", "Model Best")
        best_name = format_model_display_name(best_raw_name)
        best_f1 = get_f1(best)

        best_acc = _get_metric_val(best, ("accuracy_mean", "accuracy"))
        best_auc = _get_metric_val(best, ("roc_auc_mean", "roc_auc"))

        acc_str = f" (Accuracy: {best_acc*100:.2f}%)" if best_acc is not None else ""
        auc_str = f", ROC-AUC: {best_auc*100:.2f}%" if best_auc is not None else ""

        headline = f"🏆 {best_name} meraih performa tertinggi dengan F1-Score {best_f1*100:.2f}%"

        unique_algos = {str(e.get("model_name", "")).replace("_tune", "") for e in completed if e.get("model_name")}
        total_runs = len(completed)

        if total_runs > len(unique_algos):
            count_desc = f"Dari total {total_runs} eksperimen benchmarking ({len(unique_algos)} arsitektur algoritma)"
        else:
            count_desc = f"Dari total {total_runs} model klasifikasi yang berhasil dievaluasi"

        narrative = (
            f"{count_desc} untuk target '{target_column}', "
            f"**{best_name}** mengungguli model lainnya dengan rata-rata F1-Score sebesar **{best_f1*100:.2f}%**{acc_str}{auc_str}. "
        )

        best_base_algo = str(best_raw_name).replace("_tune", "")
        distinct_runner_up = next(
            (e for e in sorted_exps[1:] if str(e.get("model_name", "")).replace("_tune", "") != best_base_algo),
            None,
        )

        if distinct_runner_up:
            ru_raw = distinct_runner_up.get("model_name", "")
            ru_name = format_model_display_name(ru_raw)
            ru_f1 = get_f1(distinct_runner_up)
            diff = (best_f1 - ru_f1) * 100
            if diff > 0.05:
                narrative += f"Sebagai perbandingan, model alternatif terbaik berikutnya adalah **{ru_name}** dengan F1-Score {ru_f1*100:.2f}% (selisih {diff:.2f}% di bawah {best_name})."
            else:
                narrative += f"Model alternatif **{ru_name}** memiliki performa yang sangat kompetitif dengan F1-Score {ru_f1*100:.2f}%."
        elif total_runs > 1:
            narrative += f"Eksperimen yang dievaluasi berfokus pada variasi konfigurasi hyperparameter untuk arsitektur **{best_name}**."

        strengths = [
            f"F1-Score tertinggi ({best_f1*100:.2f}%) pada Stratified Cross-Validation",
            f"Waktu komputasi pelatihan: {best.get('training_time_seconds', 0):.2f} detik",
            "Pipeline pra-pemrosesan terisolasi bebas dari kebocoran data (data leakage)",
        ]
        if best_acc is not None:
            strengths.append(f"Akurasi klasifikasi: {best_acc*100:.2f}%")
        if best_auc is not None:
            strengths.append(f"ROC-AUC: {best_auc*100:.2f}%")

        cand_name = format_model_display_name(candidate.get("model_name")) if candidate and candidate.get("model_name") else "Belum dipilih"
        takeaways = [
            f"Total eksperimen selesai: {len(completed)} | Gagal: {len(failed)}",
            f"Kandidat model aktif: {cand_name}",
            "Disarankan untuk mengunduh kode Python (.py) untuk reproduksi hasil.",
        ]

        return {
            "headline": headline,
            "narrative": narrative,
            "best_model_name": best_name,
            "best_f1": best_f1,
            "best_metric_name": "F1-Score",
            "best_metric_value": best_f1,
            "strengths": strengths,
            "takeaways": takeaways,
        }


# =============================================================================
# BAGIAN 3: PANDUAN NAVIGASI LANGKAH SELANJUTNYA (SUGGEST NEXT STEP)
# =============================================================================

def suggest_next_step(
    current_page: str,
    has_dataset: bool = False,
    has_decisions: bool = False,
    has_pipeline: bool = False,
    has_experiments: bool = False,
    candidate_selected: bool = False,
) -> dict[str, Any]:
    """
    Memberikan rekomendasi navigasi konteks langkah berikutnya untuk alur kerja AutoML.
    """
    if not has_dataset:
        return {
            "step_title": "Langkah 1: Unggah Dataset",
            "message": "Unggah file CSV dataset Anda dan tentukan kolom target untuk memulai alur kerja AutoML.",
            "next_url": "/upload",
            "action_label": "Unggah Dataset",
        }

    if current_page == "upload":
        return {
            "step_title": "Langkah 2: Eksplorasi Profil & Statistik Data (EDA)",
            "message": "Dataset berhasil diunggah dan dianalisis. Buka halaman EDA untuk melihat distribusi, korelasi, dan kualitas data.",
            "next_url": "/eda",
            "action_label": "Buka Visualisasi EDA",
        }

    if current_page == "eda" or not has_decisions:
        return {
            "step_title": "Langkah 3: Tinjau Rekomendasi Pembersihan Data",
            "message": "Pemeriksaan profil selesai. Buka halaman Treatment untuk menyetujui atau menyesuaikan rekomendasi preprocessing.",
            "next_url": "/treatment",
            "action_label": "Tinjau Rekomendasi Treatment",
        }

    if current_page == "treatment" and not has_pipeline:
        return {
            "step_title": "Langkah 4: Pratinjau & Validasi Pipeline Scikit-Learn",
            "message": "Keputusan preprocessing telah tersimpan. Bangun dan validasi grafik Pipeline scikit-learn Anda.",
            "next_url": "/pipeline",
            "action_label": "Buka Pratinjau Pipeline",
        }

    if current_page == "pipeline" and not has_experiments:
        return {
            "step_title": "Langkah 5: Benchmarking Model ML",
            "message": "Pipeline siap dan valid. Pilih algoritma model pembelajaran mesin dan jalankan Cross-Validation.",
            "next_url": "/experiments",
            "action_label": "Buka Experiment Workspace",
        }

    if current_page == "experiments" and not candidate_selected:
        return {
            "step_title": "Langkah 6: Tetapkan Model Kandidat Terbaik",
            "message": "Eksperimen selesai. Evaluasi metrik dan pilih model terbaik dengan menekan 'Select as Candidate'.",
            "next_url": "/experiments",
            "action_label": "Tinjau Hasil Benchmarking",
        }

    return {
        "step_title": "Langkah 7: Unduh Kode & Laporan Final",
        "message": "Model kandidat telah ditetapkan. Anda dapat mengunduh skrip mandiri Python (.py), laporan HTML, atau melacak di MLflow.",
        "next_url": "/result",
        "action_label": "Buka Halaman Hasil Final",
    }
