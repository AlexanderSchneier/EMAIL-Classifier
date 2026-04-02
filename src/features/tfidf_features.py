"""TF-IDF word n-gram feature extractor."""

import logging
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import MAX_TFIDF_FEATURES, TFIDF_NGRAM_RANGE

logger = logging.getLogger(__name__)


class TfidfFeatures:
    """Fit-transform wrapper around TfidfVectorizer with no data leakage."""

    def __init__(
        self,
        max_features: int = MAX_TFIDF_FEATURES,
        ngram_range: tuple = TFIDF_NGRAM_RANGE,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
        )
        self._fitted = False

    def fit(self, texts):
        logger.info("Fitting TF-IDF vectorizer on %d documents...", len(texts))
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts) -> sp.csr_matrix:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts) -> sp.csr_matrix:
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
