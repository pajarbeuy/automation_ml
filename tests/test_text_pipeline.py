from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from preprocessing.text_transformers import (
    TextCleanerTransformer,
    build_text_pipeline,
    get_sentiment_stopwords,
)
from preprocessing.pipeline_builder import build_nlp_sentiment_pipeline


def test_text_cleaner_transformer():
    cleaner = TextCleanerTransformer(lowercase=True)

    raw_samples = [
        "Check this out: https://example.com/item?id=123 #promo",
        "<p>Review with <b>HTML</b> tags!</p>",
        "EXCELLENT Product!! 100% Recommended :)",
        "",  # Empty string
    ]

    cleaned = cleaner.transform(raw_samples)
    assert len(cleaned) == 4

    # URL stripped
    assert "https://" not in cleaned[0]
    assert "item?id=123" not in cleaned[0]

    # HTML tags stripped
    assert "<p>" not in cleaned[1]
    assert "<b>" not in cleaned[1]
    assert "review with html tags" in cleaned[1]

    # Lowercased
    assert "excellent product" in cleaned[2]

    # Empty string replaced with placeholder
    assert cleaned[3] == "[EMPTY]"


def test_sentiment_stopwords_preserve_negations():
    stopwords = get_sentiment_stopwords()
    # Critical negation words must NOT be removed
    assert "not" not in stopwords
    assert "no" not in stopwords
    assert "never" not in stopwords
    # Regular stopwords must be present
    assert "the" in stopwords
    assert "and" in stopwords


def test_nlp_sentiment_pipeline_zero_leakage():
    texts = [
        "great product love it",
        "awesome quality very good",
        "super fast shipping excellent",
        "horrible experience hate it",
        "terrible service bad quality",
        "worst purchase ever broken",
    ]
    labels = np.array([1, 1, 1, 0, 0, 0])
    df = pd.DataFrame({"text": texts, "target": labels})

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(df["text"], df["target"]):
        train_texts = df["text"].iloc[train_idx]
        val_texts = df["text"].iloc[val_idx]

        pipe, _ = build_nlp_sentiment_pipeline(
            text_column="text",
            model_estimator=LogisticRegression(),
            ngram_range=(1, 1),
            min_df=1,
        )

        pipe.fit(train_texts, df["target"].iloc[train_idx])

        # Vocabulary must ONLY contain tokens from train_texts
        vec = pipe.named_steps["vectorizer"]
        train_words = set(" ".join(train_texts).lower().split())

        for vocab_word in vec.vocabulary_.keys():
            assert vocab_word in train_words, f"Leakage detected: {vocab_word} not in train set"

        preds = pipe.predict(val_texts)
        assert len(preds) == len(val_idx)
