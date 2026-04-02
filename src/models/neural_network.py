"""MLP Neural Network classifier with TruncatedSVD dimensionality reduction."""

import logging

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from config.settings import SVD_COMPONENTS, RANDOM_SEED
from src.models.base_model import BaseEmailClassifier

logger = logging.getLogger(__name__)


class NeuralNetworkClassifier(BaseEmailClassifier):
    """
    MLPClassifier (3-layer feedforward network) with TruncatedSVD preprocessing.
    Uses sklearn's MLPClassifier to avoid PyTorch/TensorFlow dependencies.
    Architecture: (512, 256, 128) with ReLU activations and early stopping.

    Labels are integer-encoded internally to work around a sklearn bug with
    string labels and early_stopping in newer versions.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (512, 256, 128),
        n_components: int = SVD_COMPONENTS,
        random_state: int = RANDOM_SEED,
        max_iter: int = 200,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        )
        self._le = LabelEncoder()

    def fit(self, X_train, y_train):
        logger.info("Fitting TruncatedSVD (%d components)...", self.n_components)
        X_reduced = self.svd.fit_transform(X_train)
        y_enc = self._le.fit_transform(y_train)
        logger.info("Fitting MLP %s...", self.hidden_layer_sizes)
        self.model.fit(X_reduced, y_enc)
        logger.info("MLP training stopped after %d iterations.", self.model.n_iter_)
        return self

    def predict(self, X_test):
        X_reduced = self.svd.transform(X_test)
        y_enc = self.model.predict(X_reduced)
        return self._le.inverse_transform(y_enc)

    def predict_proba(self, X_test):
        X_reduced = self.svd.transform(X_test)
        return self.model.predict_proba(X_reduced)

    def get_params(self) -> dict:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "n_components": self.n_components,
            "max_iter": self.max_iter,
        }
