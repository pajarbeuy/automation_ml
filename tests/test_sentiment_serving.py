from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from preprocessing.pipeline_builder import build_nlp_sentiment_pipeline
from serving.inference import predict


def test_sentiment_serving_inference():
    train_texts = [
        "super satisfied with this awesome phone",
        "love the battery life and great camera",
        "terrible screen broken after two days",
        "horrible customer service totally disappointed",
    ]
    train_labels = ["positive", "positive", "negative", "negative"]

    pipe, _ = build_nlp_sentiment_pipeline(
        text_column="review",
        model_estimator=LogisticRegression(C=1.0),
        ngram_range=(1, 1),
        min_df=1,
    )
    pipe.fit(pd.DataFrame({"review": train_texts}), train_labels)

    # Single inference
    res = predict(
        pipeline=pipe,
        input_data={"review": "The camera is awesome and great"},
        task_type="sentiment_analysis",
    )
    assert len(res) == 1
    assert res[0]["prediction"] == "positive"
    assert "confidence" in res[0]
    assert "contributing_words" in res[0]
    assert len(res[0]["contributing_words"]) > 0

    # Batch inference
    batch_res = predict(
        pipeline=pipe,
        input_data=[
            {"review": "broken screen horrible"},
            {"review": "great battery life awesome"},
        ],
        task_type="sentiment_analysis",
    )
    assert len(batch_res) == 2
    assert batch_res[0]["prediction"] == "negative"
    assert batch_res[1]["prediction"] == "positive"
