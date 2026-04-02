"""Multinomial Naive Bayes classifier."""

from sklearn.naive_bayes import MultinomialNB

from src.models.base_model import BaseEmailClassifier


class NaiveBayesClassifier(BaseEmailClassifier):
    """
    MultinomialNB works on TF-IDF features only (requires non-negative values).
    Do NOT pass metadata features that may contain negative values.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.model = MultinomialNB(alpha=alpha)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def get_params(self) -> dict:
        return {"alpha": self.alpha}
