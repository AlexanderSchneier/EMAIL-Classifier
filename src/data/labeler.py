"""Map binary spam/ham labels to 5-class multi-label schema using keyword heuristics."""

import logging

import pandas as pd

from config.settings import (
    PHISHING_KEYWORDS,
    PROMOTIONS_KEYWORDS,
    SOCIAL_KEYWORDS,
)

logger = logging.getLogger(__name__)


def _contains_any(text: str, keywords: list) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _assign_label(row: pd.Series) -> str:
    """
    Heuristic multi-class label assignment.

    Spam emails → split into 'spam' vs 'phishing'
    Ham emails  → split into 'promotions', 'social', or 'personal_work'
    """
    combined = f"{row.get('subject', '')} {row.get('body', '')}"

    if row["label_raw"] == "spam":
        if _contains_any(combined, PHISHING_KEYWORDS):
            return "phishing"
        return "spam"
    else:  # ham
        if _contains_any(combined, PROMOTIONS_KEYWORDS):
            return "promotions"
        if _contains_any(combined, SOCIAL_KEYWORDS):
            return "social"
        return "personal_work"


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `label` column with one of 5 class values:
        spam, phishing, promotions, social, personal_work

    Requires columns: label_raw, subject, body
    """
    if "label_raw" not in df.columns:
        raise ValueError("DataFrame must have a 'label_raw' column (values: 'spam' or 'ham').")

    df = df.copy()
    df["label"] = df.apply(_assign_label, axis=1)
    label_distribution(df)
    return df


def label_distribution(df: pd.DataFrame) -> None:
    """Log class counts and warn on severe imbalance."""
    counts = df["label"].value_counts()
    total = len(df)
    logger.info("Label distribution:")
    for label, count in counts.items():
        pct = 100 * count / total
        logger.info("  %-20s %5d  (%.1f%%)", label, count, pct)
        if pct < 2.0:
            logger.warning("  Class '%s' has very few samples (%.1f%%) — consider oversampling.", label, pct)
