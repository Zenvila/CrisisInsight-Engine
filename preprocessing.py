"""
preprocessing.py — Text Cleaning & TF-IDF Feature Extraction
=============================================================
Person 1 (ML Engineer) module.

Handles all text preprocessing:
  - Lowercasing, punctuation removal, stopword removal, lemmatization
  - TF-IDF vectorizer fitting and transformation
  - Saving/loading vectorizer for inference
"""

import re
import os
import string
import joblib
import numpy as np

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MAX_FEATURES = 5000
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ──────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Full text-cleaning pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove numbers (standalone)
      4. Remove punctuation
      5. Tokenize
      6. Remove stopwords
      7. Lemmatize
      8. Rejoin
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove standalone numbers
    text = re.sub(r"\b\d+\b", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords + lemmatize
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]

    return " ".join(tokens)


def clean_corpus(texts: list) -> list:
    """Clean a list of texts."""
    return [clean_text(t) for t in texts]


# ──────────────────────────────────────────────
# TF-IDF Vectorizer
# ──────────────────────────────────────────────
def build_tfidf(corpus: list, max_features: int = MAX_FEATURES) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the cleaned corpus.
    Saves the fitted vectorizer to disk for later inference.

    Args:
        corpus: list of cleaned text strings
        max_features: maximum vocabulary size

    Returns:
        Fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=2,                # ignore very rare terms
        max_df=0.95,             # ignore very common terms
        sublinear_tf=True,       # apply log normalization
    )
    vectorizer.fit(corpus)

    # Save vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[OK] TF-IDF vectorizer saved -> {VECTORIZER_PATH}")
    print(f"    Vocabulary size: {len(vectorizer.vocabulary_)}")

    return vectorizer


def load_tfidf() -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer not found at {VECTORIZER_PATH}. Run train_models.py first."
        )
    return joblib.load(VECTORIZER_PATH)


def transform_text(text: str, vectorizer: TfidfVectorizer = None):
    """
    Clean and transform a single text into a TF-IDF vector.

    Args:
        text: raw text string
        vectorizer: optional pre-loaded vectorizer (loads from disk if None)

    Returns:
        Sparse TF-IDF matrix (1 × n_features)
    """
    if vectorizer is None:
        vectorizer = load_tfidf()

    cleaned = clean_text(text)
    return vectorizer.transform([cleaned])


def transform_batch(texts: list, vectorizer: TfidfVectorizer = None):
    """
    Clean and transform multiple texts into TF-IDF vectors.

    Args:
        texts: list of raw text strings
        vectorizer: optional pre-loaded vectorizer

    Returns:
        Sparse TF-IDF matrix (n_texts × n_features)
    """
    if vectorizer is None:
        vectorizer = load_tfidf()

    cleaned = clean_corpus(texts)
    return vectorizer.transform(cleaned)


def get_top_features(text: str, vectorizer: TfidfVectorizer = None, top_n: int = 10) -> list:
    """
    Get the top TF-IDF features (words) for a given text.
    Used for the explainability feature.

    Args:
        text: raw input text
        vectorizer: optional pre-loaded vectorizer
        top_n: number of top features to return

    Returns:
        List of (word, tfidf_score) tuples, sorted descending
    """
    if vectorizer is None:
        vectorizer = load_tfidf()

    tfidf_vector = transform_text(text, vectorizer)
    feature_names = vectorizer.get_feature_names_out()

    # Get non-zero entries
    dense = tfidf_vector.toarray().flatten()
    top_indices = np.argsort(dense)[::-1][:top_n]

    return [
        {"word": feature_names[i], "score": round(float(dense[i]), 4)}
        for i in top_indices
        if dense[i] > 0
    ]


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    sample = "BREAKING: Major earthquake strikes coastal region, thousands evacuated!"
    print(f"Original : {sample}")
    print(f"Cleaned  : {clean_text(sample)}")
