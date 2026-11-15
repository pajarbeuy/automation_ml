from __future__ import annotations

import collections
import re
from typing import Any
import numpy as np
import pandas as pd

# Core english and common multilingual negation-aware stopword set
COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "it", "its", "they", "them",
    "their", "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
    "this", "that", "these", "those", "which", "who", "whom", "what",
    # Indonesian common stopwords
    "yang", "di", "dan", "ke", "dari", "ini", "itu", "untuk", "pada", "adalah",
    "dengan", "ia", "mereka", "kita", "kamu", "dia", "anda", "saya", "aku",
}

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_'\-]+\b")


def profile_text(
    df: pd.DataFrame,
    text_column: str,
    target_column: str | None = None,
    max_vocab_rank: int = 25,
) -> dict[str, Any]:
    """
    Analyzes lexical characteristics, vocabulary diversity, token lengths,
    and n-gram distributions of a text column for autonomous NLP sentiment modeling.
    """
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")

    raw_series = df[text_column].fillna("").astype(str)
    doc_count = len(raw_series)

    char_lengths = raw_series.apply(len).to_numpy()
    empty_docs_count = int((raw_series.str.strip() == "").sum())

    total_tokens = 0
    token_counts = []
    unigram_counter: collections.Counter[str] = collections.Counter()
    bigram_counter: collections.Counter[str] = collections.Counter()
    stopword_count = 0
    total_uppercase_chars = 0
    total_punctuation_chars = 0

    punctuation_set = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    for text in raw_series:
        # Punctuation & case stats
        for ch in text:
            if ch.isupper():
                total_uppercase_chars += 1
            elif ch in punctuation_set:
                total_punctuation_chars += 1

        tokens = TOKEN_PATTERN.findall(text.lower())
        token_count = len(tokens)
        token_counts.append(token_count)
        total_tokens += token_count

        for i, tok in enumerate(tokens):
            unigram_counter[tok] += 1
            if tok in COMMON_STOPWORDS:
                stopword_count += 1
            if i < len(tokens) - 1:
                bigram_counter[f"{tok} {tokens[i+1]}"] += 1

    token_counts_arr = np.array(token_counts, dtype=int) if token_counts else np.array([0])
    vocab_size = len(unigram_counter)
    lexical_richness = round(vocab_size / max(total_tokens, 1), 4)
    stopword_ratio = round(stopword_count / max(total_tokens, 1), 4)

    total_chars = int(char_lengths.sum()) if len(char_lengths) > 0 else 1
    uppercase_ratio = round(total_uppercase_chars / max(total_chars, 1), 4)
    punctuation_ratio = round(total_punctuation_chars / max(total_chars, 1), 4)

    # Top unigrams
    top_unigrams = [
        {
            "token": word,
            "count": count,
            "frequency": round(count / max(total_tokens, 1), 4),
        }
        for word, count in unigram_counter.most_common(max_vocab_rank)
    ]

    # Top bigrams
    top_bigrams = [
        {
            "bigram": bigram,
            "count": count,
        }
        for bigram, count in bigram_counter.most_common(max_vocab_rank)
    ]

    # Histogram of token lengths (10 bins for visualization)
    hist_counts, bin_edges = np.histogram(token_counts_arr, bins=min(10, max(len(np.unique(token_counts_arr)), 1)))
    length_histogram = [
        {
            "bin_start": round(float(bin_edges[i]), 1),
            "bin_end": round(float(bin_edges[i + 1]), 1),
            "count": int(hist_counts[i]),
        }
        for i in range(len(hist_counts))
    ]

    # Target class distribution if target column is provided
    target_distribution: dict[str, Any] = {}
    if target_column and target_column in df.columns:
        counts = df[target_column].value_counts(dropna=False).to_dict()
        target_distribution = {
            str(k): {
                "count": int(v),
                "proportion": round(int(v) / max(doc_count, 1), 4),
            }
            for k, v in counts.items()
        }

    return {
        "text_column": text_column,
        "doc_count": doc_count,
        "empty_docs_count": empty_docs_count,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "lexical_richness": lexical_richness,
        "stopword_ratio": stopword_ratio,
        "uppercase_ratio": uppercase_ratio,
        "punctuation_ratio": punctuation_ratio,
        "char_stats": {
            "mean": round(float(char_lengths.mean()), 1) if len(char_lengths) else 0.0,
            "median": round(float(np.median(char_lengths)), 1) if len(char_lengths) else 0.0,
            "min": int(char_lengths.min()) if len(char_lengths) else 0,
            "max": int(char_lengths.max()) if len(char_lengths) else 0,
            "std": round(float(char_lengths.std()), 1) if len(char_lengths) else 0.0,
        },
        "token_stats": {
            "mean": round(float(token_counts_arr.mean()), 1),
            "median": round(float(np.median(token_counts_arr)), 1),
            "min": int(token_counts_arr.min()),
            "max": int(token_counts_arr.max()),
            "std": round(float(token_counts_arr.std()), 1),
        },
        "length_histogram": length_histogram,
        "top_unigrams": top_unigrams,
        "top_bigrams": top_bigrams,
        "target_distribution": target_distribution,
    }
