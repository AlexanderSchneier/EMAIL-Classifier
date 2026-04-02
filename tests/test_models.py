"""Tests for all 5 classifier models using tiny synthetic data."""

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.naive_bayes import NaiveBayesClassifier
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.svm import SVMClassifier
from src.models.random_forest import RandomForestEmailClassifier
from src.models.neural_network import NeuralNetworkClassifier


LABELS_6 = ["spam", "phishing", "promotions", "social", "personal", "work"]
N_SAMPLES = 120
N_FEATURES = 50


def _make_data(n_classes=6):
    """Synthetic sparse feature matrix with multi-class labels."""
    X, y_int = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_classes=n_classes,
        n_informative=n_classes * 2,
        n_redundant=2,
        random_state=42,
    )
    # Make X non-negative for NB and convert to sparse
    X = np.abs(X)
    X_sparse = sp.csr_matrix(X)
    y = np.array(LABELS_6[:n_classes])[y_int]
    split = int(0.8 * N_SAMPLES)
    return X_sparse[:split], X_sparse[split:], y[:split], y[split:]


X_train, X_test, y_train, y_test = _make_data()


# ── Helper ────────────────────────────────────────────────────────────────────

def _check_model(clf, X_tr=X_train, X_te=X_test, y_tr=y_train, y_te=y_test):
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert len(preds) == len(y_te)
    assert all(p in LABELS_6 for p in preds)
    return preds


# ── Naive Bayes ───────────────────────────────────────────────────────────────

def test_naive_bayes_predict():
    _check_model(NaiveBayesClassifier())


def test_naive_bayes_predict_proba():
    clf = NaiveBayesClassifier()
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(y_test), 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_naive_bayes_params():
    clf = NaiveBayesClassifier(alpha=0.5)
    assert clf.get_params()["alpha"] == 0.5


# ── Logistic Regression ───────────────────────────────────────────────────────

def test_logistic_regression_predict():
    _check_model(LogisticRegressionClassifier())


def test_logistic_regression_predict_proba():
    clf = LogisticRegressionClassifier()
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape[1] == 5


def test_logistic_regression_top_features():
    clf = LogisticRegressionClassifier()
    clf.fit(X_train, y_train)
    feature_names = [f"feat_{i}" for i in range(N_FEATURES)]
    clf.set_feature_names(feature_names)
    top = clf.get_top_features_per_class(n=3)
    assert len(top) == 5
    for cls, feats in top.items():
        assert len(feats) == 3


# ── SVM ───────────────────────────────────────────────────────────────────────

def test_svm_predict():
    _check_model(SVMClassifier())


def test_svm_predict_proba():
    clf = SVMClassifier()
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape[1] == 5


# ── Random Forest ─────────────────────────────────────────────────────────────

def test_random_forest_predict():
    _check_model(RandomForestEmailClassifier(n_estimators=20, n_components=10))


def test_random_forest_predict_proba():
    clf = RandomForestEmailClassifier(n_estimators=20, n_components=10)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape[1] == 5


# ── Neural Network ────────────────────────────────────────────────────────────

def test_neural_network_predict():
    clf = NeuralNetworkClassifier(
        hidden_layer_sizes=(32, 16), n_components=10, max_iter=50
    )
    _check_model(clf)


def test_neural_network_predict_proba():
    clf = NeuralNetworkClassifier(
        hidden_layer_sizes=(32, 16), n_components=10, max_iter=50
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape[1] == 5
