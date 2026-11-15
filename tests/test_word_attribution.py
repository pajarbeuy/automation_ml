from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from assistant.text_explainer import (
    extract_global_word_attributions,
    explain_sentence_sentiment,
)
from preprocessing.pipeline_builder import build_nlp_sentiment_pipeline


def test_word_attributions_and_sentence_explanation():
    train_texts = [
        "excellent wonderful fantastic experience",
        "great fast helpful friendly staff",
        "terrible horrible worst product ever",
        "broken awful useless waste of money",
    ]
    train_labels = ["positive", "positive", "negative", "negative"]

    pipe, _ = build_nlp_sentiment_pipeline(
        text_column="review",
        model_estimator=LogisticRegression(C=1.0, max_iter=200),
        ngram_range=(1, 1),
        min_df=1,
    )
    pipe.fit(train_texts, train_labels)

    # 1. Global word attributions
    global_attr = extract_global_word_attributions(pipe, top_n=5)
    assert "positive" in global_attr
    assert "negative" in global_attr
    assert len(global_attr["positive"]) > 0
    assert len(global_attr["negative"]) > 0

    pos_words = [item["word"] for item in global_attr["positive"]]
    neg_words = [item["word"] for item in global_attr["negative"]]

    assert any(w in pos_words for w in ["great", "excellent", "wonderful", "fantastic"])
    assert any(w in neg_words for w in ["terrible", "horrible", "worst", "broken"])

    # 2. Local sentence explanation
    sentence = "The staff was very friendly and helpful"
    res = explain_sentence_sentiment(pipe, sentence)
    assert res["prediction"] == "positive"
    assert res["confidence"] > 0.5
    assert len(res["contributing_words"]) > 0

    contributed_tokens = [w["word"] for w in res["contributing_words"]]
    assert "friendly" in contributed_tokens or "helpful" in contributed_tokens


def test_oov_sentence_handling():
    train_texts = ["good quality", "bad quality"]
    train_labels = ["pos", "neg"]

    pipe, _ = build_nlp_sentiment_pipeline(
        text_column="text",
        model_estimator=MultinomialNB(),
        min_df=1,
    )
    pipe.fit(train_texts, train_labels)

    # Completely alien / OOV sentence
    alien_sentence = "xyzzy123 qwerty asdfgh"
    res = explain_sentence_sentiment(pipe, alien_sentence)
    assert res["prediction"] in ("pos", "neg")
    assert isinstance(res["confidence"], float)
    assert isinstance(res["contributing_words"], list)
