"""Clean and normalize raw email text."""

import logging
import re

import nltk
import pandas as pd

from config.settings import USE_STEMMING

logger = logging.getLogger(__name__)

# Download required NLTK data once
def _ensure_nltk():
    for resource in ("stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

_STOP_WORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    text = text.lower()
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _normalize_tokens(text: str) -> str:
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    if USE_STEMMING:
        tokens = [_STEMMER.stem(t) for t in tokens]
    else:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `clean_text` column to the DataFrame.

    Input DataFrame must have a `raw_text` column.
    Returns the same DataFrame with `clean_text` added.
    """
    if "raw_text" not in df.columns:
        raise ValueError("DataFrame must have a 'raw_text' column.")

    logger.info("Preprocessing %d emails...", len(df))
    cleaned = df["raw_text"].apply(_clean_text)
    df = df.copy()
    df["clean_text"] = cleaned.apply(_normalize_tokens)
    logger.info("Preprocessing complete.")
    return df
