"""Linear SVM classifier with calibrated probabilities."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from src.models.base_model import BaseEmailClassifier


class SVMClassifier(BaseEmailClassifier):
    """
    LinearSVC wrapped in CalibratedClassifierCV for probability estimates.
    LinearSVC scales well to large sparse text matrices (O(n * features)),
    unlike kernel SVM which is O(n^2)–O(n^3).
    """

    def __init__(self, C: float = 1.0, max_iter: int = 2000, random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        base = LinearSVC(C=C, max_iter=max_iter, random_state=random_state, dual=True, class_weight="balanced")
        self.model = CalibratedClassifierCV(base, cv=3)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def get_params(self) -> dict:
        return {"C": self.C, "max_iter": self.max_iter}
