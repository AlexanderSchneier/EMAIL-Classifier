"""Random Forest classifier with TruncatedSVD dimensionality reduction."""

import logging

import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

from config.settings import SVD_COMPONENTS, RANDOM_SEED
from src.models.base_model import BaseEmailClassifier

logger = logging.getLogger(__name__)


class RandomForestEmailClassifier(BaseEmailClassifier):
    """
    Random Forest preceded by TruncatedSVD (LSA).

    Tree-based models do not benefit from high-dimensional sparse inputs.
    Reducing to SVD_COMPONENTS latent dimensions cuts memory and training
    time while preserving semantic structure.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        n_components: int = SVD_COMPONENTS,
        random_state: int = RANDOM_SEED,
    ):
        self.n_estimators = n_estimators
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X_train, y_train):
        logger.info("Fitting TruncatedSVD (%d components)...", self.n_components)
        X_reduced = self.svd.fit_transform(X_train)
        logger.info("Fitting Random Forest (%d trees)...", self.n_estimators)
        self.model.fit(X_reduced, y_train)
        self._fitted = True
        return self

    def predict(self, X_test):
        X_reduced = self.svd.transform(X_test)
        return self.model.predict(X_reduced)

    def predict_proba(self, X_test):
        X_reduced = self.svd.transform(X_test)
        return self.model.predict_proba(X_reduced)

    def get_params(self) -> dict:
        return {"n_estimators": self.n_estimators, "n_components": self.n_components}
