"""Tests for src/data/preprocessor.py"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import _clean_text, _normalize_tokens, preprocess


def test_clean_text_removes_html():
    assert "<b>" not in _clean_text("<b>Hello</b> world")


def test_clean_text_removes_urls():
    result = _clean_text("Visit https://example.com for more info")
    assert "https" not in result
    assert "example" not in result


def test_clean_text_removes_email_addresses():
    result = _clean_text("Contact us at info@example.com")
    assert "@" not in result


def test_clean_text_lowercases():
    result = _clean_text("Hello World")
    assert result == result.lower()


def test_normalize_tokens_removes_stopwords():
    # "the", "is", "a" are stopwords
    result = _normalize_tokens("the quick brown fox is a test")
    assert "the" not in result.split()
    assert "is" not in result.split()


def test_normalize_tokens_non_empty():
    result = _normalize_tokens("machine learning email classification")
    assert len(result) > 0


def test_preprocess_adds_clean_text_column():
    df = pd.DataFrame({"raw_text": ["Hello world!", "<b>Spam offer</b>"]})
    result = preprocess(df)
    assert "clean_text" in result.columns
    assert len(result) == 2


def test_preprocess_raises_without_raw_text():
    df = pd.DataFrame({"text": ["something"]})
    with pytest.raises(ValueError, match="raw_text"):
        preprocess(df)


def test_preprocess_does_not_modify_original():
    df = pd.DataFrame({"raw_text": ["test email"]})
    original_cols = list(df.columns)
    preprocess(df)
    assert list(df.columns) == original_cols
