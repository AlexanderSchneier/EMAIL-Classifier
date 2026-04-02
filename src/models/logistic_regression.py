"""Multinomial Logistic Regression classifier."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.base_model import BaseEmailClassifier


class LogisticRegressionClassifier(BaseEmailClassifier):

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state,
        )
        self._feature_names = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def set_feature_names(self, feature_names):
        self._feature_names = feature_names

    def get_top_features_per_class(self, n: int = 10) -> dict:
        """Return top-n TF-IDF tokens per class by coefficient magnitude."""
        if self._feature_names is None:
            return {}
        result = {}
        classes = self.model.classes_
        coef = self.model.coef_  # shape: (n_classes, n_features)
        for i, cls in enumerate(classes):
            top_idx = np.argsort(coef[i])[-n:][::-1]
            result[cls] = [(self._feature_names[j], coef[i][j]) for j in top_idx]
        return result

    def get_params(self) -> dict:
        return {"C": self.C, "max_iter": self.max_iter}
