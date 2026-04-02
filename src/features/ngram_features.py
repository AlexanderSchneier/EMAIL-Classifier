"""Character-level n-gram feature extractor."""

import logging
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import CHAR_NGRAM_RANGE, MAX_CHAR_FEATURES

logger = logging.getLogger(__name__)


class CharNgramFeatures:
    """Character-level n-grams — strong signal for phishing/spam regardless of word boundaries."""

    def __init__(
        self,
        max_features: int = MAX_CHAR_FEATURES,
        ngram_range: tuple = CHAR_NGRAM_RANGE,
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
        )
        self._fitted = False

    def fit(self, texts):
        logger.info("Fitting char n-gram vectorizer on %d documents...", len(texts))
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
