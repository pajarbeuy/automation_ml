from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s'!\?]")

# Negation words that must NOT be removed in sentiment analysis
NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "hardly", "scarcely",
    "tidak", "bukan", "belum", "tanpa", "kurang", "jangan",
}


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for cleaning and normalizing natural language text.
    Handles URLs, HTML tags, special symbols, casing, and whitespace.
    """

    def __init__(
        self,
        lowercase: bool = True,
        strip_urls: bool = True,
        strip_html: bool = True,
        strip_special_chars: bool = True,
        keep_punctuation: bool = True,
    ):
        self.lowercase = lowercase
        self.strip_urls = strip_urls
        self.strip_html = strip_html
        self.strip_special_chars = strip_special_chars
        self.keep_punctuation = keep_punctuation

    def fit(self, X: Any, y: Any = None) -> TextCleanerTransformer:
        return self

    def transform(self, X: Any) -> np.ndarray:
        # Convert input to 1D series of strings
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0].fillna("").astype(str)
        elif isinstance(X, pd.Series):
            series = X.fillna("").astype(str)
        elif isinstance(X, np.ndarray):
            if X.ndim > 1:
                series = pd.Series(X[:, 0]).fillna("").astype(str)
            else:
                series = pd.Series(X).fillna("").astype(str)
        elif isinstance(X, list):
            series = pd.Series(X).fillna("").astype(str)
        else:
            series = pd.Series([str(X)])

        cleaned = []
        for val in series:
            text = val
            if self.strip_urls:
                text = URL_PATTERN.sub(" ", text)
            if self.strip_html:
                text = HTML_PATTERN.sub(" ", text)
            if self.strip_special_chars:
                text = SPECIAL_CHAR_PATTERN.sub(" ", text)
            if self.lowercase:
                text = text.lower()
            text = WHITESPACE_PATTERN.sub(" ", text).strip()
            cleaned.append(text if text else "[EMPTY]")

        return np.array(cleaned, dtype=object)


def get_sentiment_stopwords(language: str = "english") -> list[str]:
    """
    Returns standard stopwords with negation words carefully pruned out
    to preserve sentiment semantics.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    base_stopwords = set(ENGLISH_STOP_WORDS)
    # Remove all negation words
    sentiment_stopwords = base_stopwords - NEGATION_WORDS
    return sorted(sentiment_stopwords)


def build_text_pipeline(
    vectorizer_type: str = "tfidf",  # "tfidf" | "count" | "sublinear_tfidf"
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int | float = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    analyzer: str = "word",
    classifier: Any = None,
) -> Pipeline:
    """
    Constructs an end-to-end scikit-learn Pipeline for text processing and classification.
    Vocabulary is guaranteed to be fitted strictly on training folds when cross-validated.
    """
    cleaner = TextCleanerTransformer(lowercase=True)

    if vectorizer_type == "count":
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            analyzer=analyzer,
            stop_words=get_sentiment_stopwords(),
        )
    else:
        # Default TF-IDF
        use_sublinear = sublinear_tf or (vectorizer_type == "sublinear_tfidf")
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=use_sublinear,
            analyzer=analyzer,
            stop_words=get_sentiment_stopwords(),
        )

    steps: list[tuple[str, Any]] = [
        ("cleaner", cleaner),
        ("vectorizer", vectorizer),
    ]

    if classifier is not None:
        steps.append(("classifier", classifier))

    return Pipeline(steps)
