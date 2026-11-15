from __future__ import annotations

import pandas as pd
import pytest
from profiling.text_profiler import profile_text


def test_text_profiler_metrics():
    data = {
        "review_text": [
            "Aplikasi ini sangat bagus dan cepat! Sangat memuaskan.",
            "Pelayanan buruk, aplikasi sering crash dan lambat.",
            "Biasa saja, fitur lumayan lengkap tapi loading lama.",
            "Hebat sekali! Recommended banget buat tim kerja.",
            "Kecewa berat, tidak ada customer service yang respon.",
            "",  # Empty doc
        ],
        "sentiment": ["positive", "negative", "neutral", "positive", "negative", "neutral"],
    }
    df = pd.DataFrame(data)

    profile = profile_text(df, text_column="review_text", target_column="sentiment")

    assert profile["doc_count"] == 6
    assert profile["empty_docs_count"] == 1
    assert profile["vocab_size"] > 15
    assert profile["total_tokens"] > 20
    assert 0.0 < profile["lexical_richness"] <= 1.0
    assert profile["char_stats"]["mean"] > 10.0
    assert profile["token_stats"]["mean"] > 3.0

    # Top unigrams and bigrams
    assert len(profile["top_unigrams"]) > 0
    assert len(profile["top_bigrams"]) > 0

    # Target class distribution
    dist = profile["target_distribution"]
    assert "positive" in dist
    assert "negative" in dist
    assert "neutral" in dist
    assert dist["positive"]["count"] == 2


def test_text_profiler_missing_column():
    df = pd.DataFrame({"col_a": [1, 2, 3]})
    with pytest.raises(ValueError, match="not found"):
        profile_text(df, text_column="review_text")
