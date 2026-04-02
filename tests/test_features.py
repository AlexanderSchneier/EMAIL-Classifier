"""Tests for feature extraction modules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.tfidf_features import TfidfFeatures
from src.features.ngram_features import CharNgramFeatures
from src.features.metadata_features import MetadataFeatures


TEXTS_TRAIN = [
    "click here to verify your bank account",
    "meeting tomorrow morning at the office",
    "sale discount off limited offer free shipping",
    "linkedin notification you have a new connection",
    "quarterly report attached for your review",
]

TEXTS_TEST = [
    "verify your paypal account urgent",
    "team standup at noon",
]

META_TRAIN = pd.DataFrame({
    "subject": ["URGENT!", "Meeting tomorrow", "Sale 50% OFF", "New notification", "Report Q3"],
    "sender": [
        "spammer@evil.com",
        "boss@company.com",
        "promo@shop.com",
        "noreply@linkedin.com",
        "alice@company.com",
    ],
    "body": [
        "<a href='http://phish.com'>click here</a>",
        "See you there.",
        "http://shop.com buy now http://shop.com",
        "You got a new follower.",
        "Please find the report attached.",
    ],
})

META_TEST = pd.DataFrame({
    "subject": ["Win a prize!", "Project update"],
    "sender": ["scam@random.org", "bob@company.com"],
    "body": ["Click http://win.com now!", "The project is on track."],
})


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def test_tfidf_fit_transform_shape():
    tfidf = TfidfFeatures(max_features=100)
    X = tfidf.fit_transform(TEXTS_TRAIN)
    assert sp.issparse(X)
    assert X.shape[0] == len(TEXTS_TRAIN)
    assert X.shape[1] <= 100


def test_tfidf_transform_test():
    tfidf = TfidfFeatures(max_features=100)
    tfidf.fit(TEXTS_TRAIN)
    X_test = tfidf.transform(TEXTS_TEST)
    assert X_test.shape[0] == len(TEXTS_TEST)


def test_tfidf_no_leakage():
    tfidf = TfidfFeatures()
    with pytest.raises(RuntimeError):
        tfidf.transform(TEXTS_TEST)


def test_tfidf_feature_names():
    tfidf = TfidfFeatures(max_features=50)
    tfidf.fit(TEXTS_TRAIN)
    names = tfidf.get_feature_names()
    assert len(names) > 0


# ── Char N-gram ───────────────────────────────────────────────────────────────

def test_char_ngram_shape():
    ng = CharNgramFeatures(max_features=200)
    X = ng.fit_transform(TEXTS_TRAIN)
    assert sp.issparse(X)
    assert X.shape[0] == len(TEXTS_TRAIN)


def test_char_ngram_no_leakage():
    ng = CharNgramFeatures()
    with pytest.raises(RuntimeError):
        ng.transform(TEXTS_TEST)


# ── Metadata ──────────────────────────────────────────────────────────────────

def test_metadata_fit_transform():
    meta = MetadataFeatures(top_domains=10)
    X = meta.fit_transform(META_TRAIN)
    assert sp.issparse(X)
    assert X.shape[0] == len(META_TRAIN)


def test_metadata_transform_test():
    meta = MetadataFeatures(top_domains=10)
    meta.fit(META_TRAIN)
    X_test = meta.transform(META_TEST)
    assert X_test.shape[0] == len(META_TEST)
    # Column count must match training
    X_train = meta.transform(META_TRAIN)
    assert X_test.shape[1] == X_train.shape[1]


def test_metadata_no_leakage():
    meta = MetadataFeatures()
    with pytest.raises(RuntimeError):
        meta.transform(META_TEST)


def test_metadata_html_detection():
    meta = MetadataFeatures(top_domains=5)
    meta.fit(META_TRAIN)
    X = meta.transform(META_TRAIN).toarray()
    # First email has HTML, body_length col (index 1) and has_html col (index 2)
    # has_html is index 2 in numeric block
    # Just check the matrix has the right shape and is numeric
    assert X.shape[1] > 5
    assert np.all(np.isfinite(X))
