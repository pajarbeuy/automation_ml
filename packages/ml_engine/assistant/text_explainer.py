from __future__ import annotations

import re
from typing import Any
import numpy as np
from sklearn.pipeline import Pipeline

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_'\-]+\b")


def _find_step(pipeline: Pipeline, step_name: str) -> Any:
    if hasattr(pipeline, "named_steps") and step_name in pipeline.named_steps:
        return pipeline.named_steps[step_name]

    # Check inside preprocessor ColumnTransformer
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        prep = pipeline.named_steps["preprocessor"]
        if hasattr(prep, "transformers_"):
            for _, trans, _ in prep.transformers_:
                if hasattr(trans, "named_steps") and step_name in trans.named_steps:
                    return trans.named_steps[step_name]
                if step_name == "vectorizer" and hasattr(trans, "vocabulary_"):
                    return trans
    return None


def extract_global_word_attributions(
    pipeline: Pipeline,
    top_n: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    """
    Extracts top positive and top negative contributing n-grams/words
    from a trained sentiment analysis pipeline.
    """
    vectorizer = _find_step(pipeline, "vectorizer")
    model = _find_step(pipeline, "model")

    if not vectorizer or not model:
        return {"positive": [], "negative": []}

    if not hasattr(vectorizer, "get_feature_names_out"):
        return {"positive": [], "negative": []}

    feature_names = vectorizer.get_feature_names_out()
    classes = getattr(model, "classes_", [0, 1])

    weights: np.ndarray | None = None
    if hasattr(model, "coef_"):
        raw_coef = model.coef_
        weights = raw_coef[0] if raw_coef.ndim > 1 and raw_coef.shape[0] == 1 else raw_coef.mean(axis=0)
    elif hasattr(model, "feature_log_prob_"):
        log_prob = model.feature_log_prob_
        if log_prob.shape[0] >= 2:
            # Positive class (1) log odds minus Negative class (0) log odds
            weights = log_prob[1] - log_prob[0]
        else:
            weights = log_prob[0]

    if weights is None or len(weights) != len(feature_names):
        return {"positive": [], "negative": []}

    # Sort indices
    sorted_idx = np.argsort(weights)

    # Top negative words (smallest weights)
    neg_idx = sorted_idx[:top_n]
    negative_words = [
        {
            "word": str(feature_names[i]),
            "weight": round(float(weights[i]), 4),
            "sentiment": "negative",
        }
        for i in neg_idx
        if weights[i] < 0
    ]

    # Top positive words (largest weights)
    pos_idx = sorted_idx[::-1][:top_n]
    positive_words = [
        {
            "word": str(feature_names[i]),
            "weight": round(float(weights[i]), 4),
            "sentiment": "positive",
        }
        for i in pos_idx
        if weights[i] > 0
    ]

    return {
        "positive": positive_words,
        "negative": negative_words,
    }


def explain_sentence_sentiment(
    pipeline: Pipeline,
    text: str,
    top_k: int = 15,
) -> dict[str, Any]:
    """
    Computes sentiment prediction, confidence score, and token-level attribution highlights
    for an individual sentence.
    """
    vectorizer = _find_step(pipeline, "vectorizer")
    model = _find_step(pipeline, "model")

    if not text or not text.strip():
        return {
            "prediction": "neutral",
            "confidence": 0.5,
            "contributing_words": [],
        }

    # Predict class
    pred = pipeline.predict([text])[0]
    classes = getattr(model, "classes_", ["negative", "positive"])

    # Predict confidence
    confidence = 0.85
    if hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba([text])[0]
            pred_idx = np.where(classes == pred)[0]
            if len(pred_idx):
                confidence = float(probas[pred_idx[0]])
            else:
                confidence = float(np.max(probas))
        except Exception:
            confidence = 0.85
    elif hasattr(pipeline, "decision_function"):
        try:
            decision = float(pipeline.decision_function([text])[0])
            confidence = round(1.0 / (1.0 + np.exp(-abs(decision))), 3)
        except Exception:
            confidence = 0.85

    # Token attribution lookup
    tokens = TOKEN_PATTERN.findall(text.lower())
    contributing: list[dict[str, Any]] = []

    if vectorizer and model and hasattr(vectorizer, "vocabulary_"):
        vocab = vectorizer.vocabulary_
        weights = None
        if hasattr(model, "coef_"):
            raw_coef = model.coef_
            weights = raw_coef[0] if raw_coef.ndim > 1 and raw_coef.shape[0] == 1 else raw_coef.mean(axis=0)
        elif hasattr(model, "feature_log_prob_"):
            log_prob = model.feature_log_prob_
            if log_prob.shape[0] >= 2:
                weights = log_prob[1] - log_prob[0]
            else:
                weights = log_prob[0]

        if weights is not None:
            seen_tokens: set[str] = set()
            for tok in tokens:
                if tok in vocab and tok not in seen_tokens:
                    idx = vocab[tok]
                    if idx < len(weights):
                        w = float(weights[idx])
                        contributing.append({
                            "word": tok,
                            "weight": round(w, 4),
                            "sentiment": "positive" if w > 0 else "negative",
                        })
                        seen_tokens.add(tok)

            # Sort by absolute impact
            contributing.sort(key=lambda x: abs(x["weight"]), reverse=True)
            contributing = contributing[:top_k]

    return {
        "prediction": str(pred),
        "confidence": round(float(confidence), 4),
        "contributing_words": contributing,
    }
