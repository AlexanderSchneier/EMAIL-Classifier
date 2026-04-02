"""Abstract base class for all email classifiers."""

from abc import ABC, abstractmethod


class BaseEmailClassifier(ABC):

    @abstractmethod
    def fit(self, X_train, y_train):
        """Train the model."""

    @abstractmethod
    def predict(self, X_test):
        """Return predicted class labels."""

    def predict_proba(self, X_test):
        """Return class probabilities. Override if supported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support predict_proba.")

    def get_params(self) -> dict:
        """Return model hyperparameters for logging."""
        return {}
