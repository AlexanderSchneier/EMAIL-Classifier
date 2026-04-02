"""Structural and sender-based metadata features."""

import logging
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

from config.settings import TOP_SENDER_DOMAINS

logger = logging.getLogger(__name__)

_EMAIL_DOMAIN_RE = re.compile(r"@([\w.\-]+)")


def _extract_domain(sender: str) -> str:
    match = _EMAIL_DOMAIN_RE.search(sender)
    return match.group(1).lower() if match else "unknown"


class MetadataFeatures:
    """
    Extracts numeric/categorical features from email headers and structure.

    Features:
        - subject_length       : character count of subject line
        - body_length          : character count of body
        - has_html             : 1 if body contains HTML tags, else 0
        - exclamation_count    : number of '!' in subject
        - num_links            : occurrences of 'http' in body
        - sender_domain_*      : one-hot for top-N sender domains (train-only fit)
        - sender_frequency     : normalised send-frequency from training set
    """

    def __init__(self, top_domains: int = TOP_SENDER_DOMAINS):
        self.top_domains = top_domains
        self._domain_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=True
        )
        self._top_domain_list = []
        self._domain_freq_map: dict = {}
        self._fitted = False

    # ── Numeric features (no fit needed) ──────────────────────────────────────

    @staticmethod
    def _numeric_features(df: pd.DataFrame) -> np.ndarray:
        subject = df["subject"].fillna("").astype(str)
        body = df["body"].fillna("").astype(str)

        subject_len = subject.str.len().values.reshape(-1, 1)
        body_len = body.str.len().values.reshape(-1, 1)
        has_html = body.str.contains(r"<[^>]+>", regex=True).astype(int).values.reshape(-1, 1)
        exclaim = subject.str.count("!").values.reshape(-1, 1)
        num_links = body.str.count("http").values.reshape(-1, 1)

        return np.hstack([subject_len, body_len, has_html, exclaim, num_links]).astype(float)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame):
        sender = df["sender"].fillna("").astype(str)
        domains = sender.apply(_extract_domain)

        # Top-N domains for one-hot encoding
        counts = domains.value_counts()
        self._top_domain_list = counts.head(self.top_domains).index.tolist()
        self._domain_encoder.fit(domains.values.reshape(-1, 1))

        # Sender frequency map (normalised by max count)
        freq = sender.value_counts()
        max_freq = freq.max() if len(freq) > 0 else 1
        self._domain_freq_map = (freq / max_freq).to_dict()

        self._fitted = True
        logger.info(
            "MetadataFeatures fitted. Top domains: %d, unique senders: %d",
            len(self._top_domain_list),
            len(self._domain_freq_map),
        )
        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        numeric = self._numeric_features(df)

        sender = df["sender"].fillna("").astype(str)
        domains = sender.apply(_extract_domain).values.reshape(-1, 1)
        domain_ohe = self._domain_encoder.transform(domains)

        freq = sender.map(self._domain_freq_map).fillna(0).values.reshape(-1, 1)

        numeric_sparse = sp.csr_matrix(np.hstack([numeric, freq]))
        return sp.hstack([numeric_sparse, domain_ohe], format="csr")

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        self.fit(df)
        return self.transform(df)
