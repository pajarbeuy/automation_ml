from __future__ import annotations

# =============================================================================
# MODUL: PIPELINE BUILDER (PEMBUAT PIPELINE PREPROCESSING & MODEL)
# =============================================================================
# Deskripsi: Modul ini bertanggung jawab mengubah keputusan user (WorkflowDecisions)
#            menjadi Pipeline Scikit-Learn yang aman dari kebocoran data (Data Leakage).
#
# Prinsip Utama (Berdasarkan PRD & Spesifikasi Teknis):
#   Dataset -> Train/Test Split -> Fit Preprocessing HANYA pada Train -> Transform Train & Test
#   Dilarang melakukan fit preprocessing pada seluruh dataset sebelum split!
# =============================================================================

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from preprocessing.transformers import FrequencyEncoder, Winsorizer
from preprocessing.text_transformers import TextCleanerTransformer, get_sentiment_stopwords

# =============================================================================
# BAGIAN 1: PEMETAAN AKSI USER (DECISION) KE OBJEK SCIKIT-LEARN
# =============================================================================

# 1. Pemetaan Imputasi Nilai Hilang (Missing Values)
IMPUTER_MAP: dict[str, Any] = {
    "mean_imputation": SimpleImputer(strategy="mean"),
    "median_imputation": SimpleImputer(strategy="median"),
    "most_frequent_imputation": SimpleImputer(strategy="most_frequent"),
    "constant_imputation": SimpleImputer(strategy="constant", fill_value="missing"),
    "knn_imputation": KNNImputer(n_neighbors=5),
    "drop_rows": None,      # Ditangani sebelum masuk ke pipeline ML
    "drop_column": None,    # Kolom akan dikecualikan dari ColumnTransformer
}

# 2. Pemetaan Penyesuaian Skala (Feature Scaling)
SCALER_MAP: dict[str, Any] = {
    "standard_scaling": StandardScaler(),
    "min_max_scaling": MinMaxScaler(),
    "robust_scaling": RobustScaler(),
    "no_scaling": "passthrough",
    "leave_unchanged": "passthrough",
}

# 3. Pemetaan Pengkodean Kategori (Categorical Encoding)
ENCODER_MAP: dict[str, Any] = {
    "one_hot_encoding": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal_encoding": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "frequency_encoding": FrequencyEncoder(),
}

# 4. Pemetaan Penanganan Outlier
OUTLIER_MAP: dict[str, Any] = {
    "winsorization": Winsorizer(factor=1.5),
    "iqr_filtering": None,     # Row-level filtering, ditangani sebelum pipeline
    "robust_scaling": None,    # Ditangani oleh scaler
    "leave_unchanged": "passthrough",
}


def _clone_step(obj: Any) -> Any:
    """
    Fungsi Pembantu: Membuat salinan independen (clone) dari objek estimator Scikit-Learn.
    Mencegah efek samping (side-effects) ketika transformer digunakan ulang di multiple pipeline.
    """
    from sklearn.base import clone
    if obj is None or obj == "passthrough":
        return obj
    return clone(obj)


# =============================================================================
# BAGIAN 2: FUNGSI INDEXING KEPUTUSAN (DECISION HELPER)
# =============================================================================

def _index_decisions(decisions: list[dict]) -> dict[str, dict[str, str]]:
    """
    Membangun kamus pencarian cepat per-kolom dan per-kategori:
    Hasil: { nama_kolom: { kategori_aksi: aksi_terpilih } }
    Contoh: { "age": { "missing_value": "median_imputation", "scaling": "standard_scaling" } }
    """
    idx: dict[str, dict[str, str]] = {}
    for d in decisions:
        col = d.get("column_name") or "__dataset__"
        cat = d["category"]
        action = d["selected_action"]
        if col not in idx:
            idx[col] = {}
        idx[col][cat] = action
    return idx


# =============================================================================
# BAGIAN 3: PENGONSTRUKSI PIPELINE (PIPELINE BUILDER CORE)
# =============================================================================

def build_pipeline(
    decisions: list[dict],
    profile_features: list,
    model_estimator: Any,
    enable_feature_engineering: bool = False,
) -> tuple[Pipeline, dict]:
    """
    Membangun Pipeline Scikit-Learn yang aman dari data leakage berdasarkan keputusan yang disetujui.

    Parameter:
        decisions: Daftar dict WorkflowDecision (hanya yang disetujui / di-override).
        profile_features: Daftar objek FeatureProfile dari profil dataset.
        model_estimator: Model ML Scikit-Learn (misal: RandomForestClassifier).
        enable_feature_engineering: True jika ingin menyertakan AutoFeatureEngineer.

    Return:
        (pipeline, config_json) -> Pipeline belum di-fit (unfitted).
    """
    # 1. Buat indeks pencarian keputusan user
    idx = _index_decisions(decisions)

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    text_cols: list[str] = []
    drop_cols: set[str] = set()
    # 2. Pengelompokan jenis kolom & pengidentifikasian kolom yang dibuang (drop)
    for f in profile_features:
        col_decisions = idx.get(f.name, {})

        missing_action = col_decisions.get("missing_value", "median_imputation" if f.dtype == "numeric" else "most_frequent_imputation")
        outlier_action = col_decisions.get("outlier", "leave_unchanged")
        encoding_action = col_decisions.get("encoding", "one_hot_encoding")
        scaling_action = col_decisions.get("scaling", "standard_scaling")

        # Jika aksi user memilih drop_column pada kategori APAPUN (data_leakage, missing_value, encoding, dll)
        if any(action == "drop_column" for action in col_decisions.values()):
            drop_cols.add(f.name)
            continue

        # Klasifikasikan fitur ke numerik, kategorikal, atau teks
        if f.dtype == "numeric" and not f.constant:
            numeric_cols.append(f.name)
        elif f.dtype in ("categorical", "boolean") and not f.constant:
            categorical_cols.append(f.name)
        elif f.dtype == "text" and not f.constant:
            text_cols.append(f.name)
        # Catatan: Kolom konstanta, datetime, atau ID dilompati
    transformers = []
    config_json: dict[str, Any] = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "text_columns": text_cols,
        "dropped_columns": list(drop_cols),
        "numeric_config": {},
        "categorical_config": {},
        "text_config": {},
        "enable_feature_engineering": enable_feature_engineering,
    }
    # 3. Membangun Sub-Pipeline untuk Kolom Numerik
    if numeric_cols:
        for col in numeric_cols:
            col_dec = idx.get(col, {})
            col_missing = col_dec.get("missing_value", "median_imputation")
            col_outlier = col_dec.get("outlier", "leave_unchanged")
            col_scaling = col_dec.get("scaling", "standard_scaling")

            steps = []
            # Langkah A: Imputasi
            imputer = IMPUTER_MAP.get(col_missing)
            if imputer is not None:
                steps.append(("imputer", _clone_step(imputer)))

            # Langkah B: Winsorization (Outlier)
            if col_outlier == "winsorization":
                steps.append(("winsorizer", Winsorizer()))

            # Langkah C: Feature Scaling
            scaler = SCALER_MAP.get(col_scaling, "passthrough")
            if scaler != "passthrough" and scaler is not None:
                steps.append(("scaler", _clone_step(scaler)))

            # Masukkan sub-pipeline kolom numerik ini ke transformer list
            if steps:
                col_pipe = Pipeline(steps)
                transformers.append((f"num_{col}", col_pipe, [col]))
            else:
                transformers.append((f"num_{col}", "passthrough", [col]))

            config_json["numeric_config"][col] = {
                "missing": col_missing,
                "outlier": col_outlier,
                "scaling": col_scaling,
            }

    # 4. Membangun Sub-Pipeline untuk Kolom Kategorikal
    if categorical_cols:
        for col in categorical_cols:
            col_dec = idx.get(col, {})
            col_missing = col_dec.get("missing_value", "most_frequent_imputation")
            col_encoding = col_dec.get("encoding", "one_hot_encoding")

            steps = []
            # Langkah A: Imputasi Nilai Hilang
            imputer = IMPUTER_MAP.get(col_missing)
            if imputer is not None:
                steps.append(("imputer", _clone_step(imputer)))

            # Langkah B: Categorical Encoding
            encoder = ENCODER_MAP.get(col_encoding)
            if encoder is not None:
                steps.append(("encoder", _clone_step(encoder)))

            # Masukkan sub-pipeline kolom kategorikal ini ke transformer list
            if steps:
                col_pipe = Pipeline(steps)
                transformers.append((f"cat_{col}", col_pipe, [col]))
            else:
                transformers.append((f"cat_{col}", "passthrough", [col]))

            config_json["categorical_config"][col] = {
                "missing": col_missing,
                "encoding": col_encoding,
            }

    # 4b. Membangun Sub-Pipeline untuk Kolom Teks (NLP / Sentimen)
    if text_cols:
        for col in text_cols:
            col_pipe = Pipeline([
                ("cleaner", TextCleanerTransformer(lowercase=True)),
                ("vectorizer", TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    stop_words=get_sentiment_stopwords(),
                )),
            ])
            transformers.append((f"text_{col}", col_pipe, col))
            config_json["text_config"][col] = {
                "vectorizer": "tfidf",
                "ngram_range": [1, 2],
                "min_df": 2,
                "sublinear_tf": True,
            }

    # 5. Menggabungkan Semua Preprocessing dengan ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # 6. Membunyikan Pipeline Akhir (Preprocessor + Feature Engineer + Estimator ML)
    pipeline_steps = [("preprocessor", preprocessor)]
    if enable_feature_engineering:
        from preprocessing.feature_engineering import AutoFeatureEngineer
        pipeline_steps.append(("feature_engineer", AutoFeatureEngineer()))
    pipeline_steps.append(("model", model_estimator))

    pipeline = Pipeline(pipeline_steps)

    config_json["model"] = {
        "name": type(model_estimator).__name__,
        "params": model_estimator.get_params(),
    }

    return pipeline, config_json


# =============================================================================
# BAGIAN 4: FUNGSI PENDUKUNG (HASHING & PREVIEW DESCRIPTION)
# =============================================================================

def config_hash(config_json: dict) -> str:
    """
    Menghasilkan nilai hash SHA256 unik dari konfigurasi pipeline.
    Berguna untuk memverifikasi apakah pipeline yang sama pernah dibuat sebelumnya.
    """
    serialized = json.dumps(config_json, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def get_pipeline_steps_description(pipeline: Pipeline) -> list[str]:
    """
    Mengembalikan daftar penjelasan langkah-langkah pipeline yang mudah dibaca manusia.
    Dapat dijalankan pada pipeline yang belum di-fit (unfitted).
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    steps = []
    for name, transformer, columns in preprocessor.transformers:
        if transformer in ("drop", "passthrough"):
            steps.append(f"{name}: {transformer} ({len(columns)} col(s))")
        elif hasattr(transformer, "steps"):
            sub = " -> ".join(sname for sname, _ in transformer.steps)
            steps.append(f"{name}: {sub} ({len(columns)} col(s))")
        else:
            steps.append(f"{name}: {type(transformer).__name__} ({len(columns)} col(s))")
    model_name = type(pipeline.named_steps["model"]).__name__
    steps.append(f"model: {model_name}")
    return steps



def build_nlp_sentiment_pipeline(
    text_column: str,
    model_estimator: Any,
    vectorizer_type: str = "tfidf",
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int | float = 2,
    sublinear_tf: bool = True,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Construct a dedicated NLP sentiment pipeline for a single text column.
    Guarantees strict zero-leakage fold fitting.
    """
    cleaner = TextCleanerTransformer(lowercase=True)
    if vectorizer_type == "count":
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words=get_sentiment_stopwords(),
        )
    else:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            stop_words=get_sentiment_stopwords(),
        )

    pipeline = Pipeline([
        ("cleaner", cleaner),
        ("vectorizer", vectorizer),
        ("model", model_estimator),
    ])

    config_json = {
        "task_type": "sentiment_analysis",
        "text_column": text_column,
        "vectorizer": vectorizer_type,
        "ngram_range": list(ngram_range),
        "min_df": min_df,
        "sublinear_tf": sublinear_tf,
        "model": {
            "name": type(model_estimator).__name__,
            "params": model_estimator.get_params(),
        },
    }
    return pipeline, config_json

