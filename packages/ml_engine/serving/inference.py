from __future__ import annotations

# =============================================================================
# MODUL: MODEL SERVING & INFERENCE ENGINE
# =============================================================================
# Deskripsi: Menjalankan inferensi prediksi langsung (live prediction) menggunakan
#            artefak pipeline.pkl scikit-learn yang telah dilatih.
#
# Keistimewaan:
# - Menerima data input mentah (raw features) tanpa perlu scaling/encoding manual.
# - Preprocessing (imputer, winsorizer, encoder, scaler, feature engineer) berjalan
#   secara otomatis dan transparan di dalam pipeline.
# - Mendukung single-row dict, batch list of dicts, maupun pandas DataFrame.
# - Mengembalikan probabilitas kelas & confidence score untuk Klasifikasi.
# - Mengembalikan nilai prediksi kontinu untuk Regresi.
# =============================================================================

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd


def load_pipeline(pipeline_path: str) -> Any:
    """Memuat pipeline Scikit-Learn dari file pickle."""
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"File artefak pipeline tidak ditemukan pada: {pipeline_path}")
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def predict(
    pipeline: Any,
    input_data: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    task_type: str = "classification",
    metrics: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Menjalankan inferensi model dari input data mentah (dict, list of dict, atau DataFrame).
    Mengembalikan list of dict berisi hasil prediksi, keyakinan (confidence),
    serta probabilitas kelas (untuk klasifikasi) atau rentang estimasi (untuk regresi).
    """
    if isinstance(input_data, dict):
        if not input_data:
            raise ValueError("Data input inferensi tidak boleh kosong.")
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        if not input_data:
            raise ValueError("Data input inferensi tidak boleh kosong.")
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        if input_data.empty:
            raise ValueError("Data input inferensi tidak boleh kosong.")
        df = input_data.copy()
    else:
        raise ValueError(
            f"Tipe data input tidak didukung: {type(input_data)}. Gunakan dict, list of dict, atau DataFrame."
        )

    # 1. Deteksi kolom fitur yang dibutuhkan model
    required_features = None
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        prep = pipeline.named_steps["preprocessor"]
        transformers_list = getattr(prep, "transformers_", None) or getattr(prep, "transformers", None)
        if transformers_list:
            req = []
            for name, trans, cols in transformers_list:
                # Abaikan remainder atau transformer 'drop'
                if trans != "drop" and name != "remainder":
                    if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                        req.extend([str(c) for c in cols])
                    elif isinstance(cols, str):
                        req.append(cols)
            if req:
                required_features = req

    # Fallback ke feature_names_in_ jika preprocessor bukan ColumnTransformer
    if required_features is None:
        raw_exp = getattr(pipeline, "feature_names_in_", None)
        if raw_exp is None and hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
            raw_exp = getattr(pipeline.named_steps["preprocessor"], "feature_names_in_", None)
        if raw_exp is not None:
            required_features = [str(col) for col in raw_exp]

    if required_features is not None:
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input inferensi tidak lengkap. Kolom fitur berikut wajib disertakan: {missing_cols}"
            )

    # 2. Isi kolom yang di-drop / tidak terpakai dengan NaN jika pipeline merekam feature_names_in_
    pipeline_features = getattr(pipeline, "feature_names_in_", None)
    if pipeline_features is not None:
        for col in pipeline_features:
            if col not in df.columns:
                df[col] = np.nan

    try:
        predictions = pipeline.predict(df)
    except ValueError as exc:
        msg = str(exc)
        if "could not convert string to float" in msg:
            raise ValueError(f"Tipe data tidak valid: nilai numerik tidak boleh berupa teks ({msg})") from exc
        raise ValueError(f"Gagal menjalankan inferensi model: {msg}") from exc
    except Exception as exc:
        raise ValueError(f"Gagal menjalankan inferensi model: {exc}") from exc

    has_proba = False
    probabilities = None
    classes = None
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(df)
            has_proba = True
            classes = getattr(pipeline, "classes_", None)
            if classes is None and hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
                classes = getattr(pipeline.named_steps["model"], "classes_", None)
        except Exception:
            has_proba = False

    results: list[dict[str, Any]] = []
    for i, pred in enumerate(predictions):
        if task_type == "regression":
            pred_val = float(pred)
            row_res: dict[str, Any] = {"prediction": round(pred_val, 4)}
            if metrics:
                r2 = metrics.get("r2_mean") if "r2_mean" in metrics else metrics.get("r2")
                mae = metrics.get("mae_mean") if "mae_mean" in metrics else metrics.get("mae")
                if r2 is not None:
                    row_res["confidence"] = round(max(0.0, min(0.9999, float(r2))), 4)
                    row_res["confidence_metric"] = "R2 Score"
                if mae is not None:
                    mae_val = float(mae)
                    row_res["margin_of_error"] = round(mae_val, 2)
                    row_res["interval_lower"] = round(max(0.0, pred_val - mae_val), 2)
                    row_res["interval_upper"] = round(pred_val + mae_val, 2)
            results.append(row_res)
        else:
            pred_label = pred.item() if hasattr(pred, "item") else pred
            row_res = {"prediction": pred_label}
            if has_proba and probabilities is not None:
                row_probs = probabilities[i]
                confidence = float(np.max(row_probs))
                row_res["confidence"] = round(confidence, 4)
                row_res["confidence_metric"] = "Class Probability"
                if classes is not None and len(classes) == len(row_probs):
                    row_res["probabilities"] = {
                        str(cls): round(float(p), 4)
                        for cls, p in zip(classes, row_probs)
                    }
            results.append(row_res)

    return results
