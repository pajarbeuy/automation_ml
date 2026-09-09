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
) -> list[dict[str, Any]]:
    """
    Menjalankan inferensi prediksi pada data fitur mentah.

    Parameter:
        pipeline: Objek Scikit-Learn Pipeline yang sudah di-fit.
        input_data: Dict single-row, list of dicts, atau pandas DataFrame.
        task_type: 'classification' atau 'regression'.

    Return:
        List of dicts hasil prediksi untuk setiap baris data input:
        - Klasifikasi: {"prediction": class_label, "confidence": float, "probabilities": {...}}
        - Regresi: {"prediction": float}
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("input_data harus berupa dict, list of dicts, atau pandas DataFrame")

    predictions = pipeline.predict(df)

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
            results.append({"prediction": round(pred_val, 4)})
        else:
            pred_label = pred.item() if hasattr(pred, "item") else pred
            row_res: dict[str, Any] = {"prediction": pred_label}
            if has_proba and probabilities is not None:
                row_probs = probabilities[i]
                confidence = float(np.max(row_probs))
                row_res["confidence"] = round(confidence, 4)
                if classes is not None and len(classes) == len(row_probs):
                    row_res["probabilities"] = {
                        str(cls): round(float(p), 4)
                        for cls, p in zip(classes, row_probs)
                    }
            results.append(row_res)

    return results
