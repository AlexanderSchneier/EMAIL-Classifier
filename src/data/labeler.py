"""Map raw labels to 6-class schema using CMU annotations where available, keywords as fallback."""

import logging

import pandas as pd

from config.settings import (
    CMU_LABEL_MAP,
    PERSONAL_KEYWORDS,
    PHISHING_KEYWORDS,
    PROMOTIONS_KEYWORDS,
    SOCIAL_KEYWORDS,
    WORK_KEYWORDS,
)

logger = logging.getLogger(__name__)

# CMU annotation columns in the brianray dataset.
# Each annotation slot has level_1 (coarse category), level_2 (subcategory), weight.
_N_CAT_SLOTS = 12


def _contains_any(text: str, keywords: list) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _label_from_cmu(row: pd.Series) -> str | None:
    """
    Return a label derived from CMU annotation columns, or None if no
    annotation is present.

    Columns follow the pattern: cat_i_level_1, cat_i_level_2, cat_i_weight
    When multiple slots are populated, the one with the highest weight wins.
    level_1 and level_2 are floats (e.g. 1.0, 1.0) → code "1.1".
    """
    best_weight = -1.0
    best_code = None

    for i in range(1, _N_CAT_SLOTS + 1):
        l1_col = f"cat_{i}_level_1"
        l2_col = f"cat_{i}_level_2"
        w_col  = f"cat_{i}_weight"

        if l1_col not in row.index:
            continue

        l1 = row.get(l1_col)
        l2 = row.get(l2_col)
        if pd.isna(l1) or pd.isna(l2):
            continue

        try:
            code = f"{int(float(l1))}.{int(float(l2))}"
        except (ValueError, TypeError):
            continue

        if code not in CMU_LABEL_MAP:
            continue

        weight = float(row.get(w_col, 0) or 0)
        if weight > best_weight:
            best_weight = weight
            best_code = code

    return CMU_LABEL_MAP[best_code] if best_code else None


def _label_from_keywords(row: pd.Series) -> str:
    """
    Keyword heuristic fallback for unlabeled records.

    Nazario phishing emails carry label_raw='phishing' and are passed through directly.
    SpamAssassin spam is split into spam/phishing by keyword.
    Ham is split into work/personal by keyword.
    """
    combined = f"{row.get('subject', '')} {row.get('body', '')}"

    # Nazario corpus emails are already definitively labeled
    if row["label_raw"] == "phishing":
        return "phishing"

    if row["label_raw"] == "spam":
        if _contains_any(combined, PHISHING_KEYWORDS):
            return "phishing"
        return "spam"

    # ham — check in priority order
    if _contains_any(combined, PROMOTIONS_KEYWORDS):
        return "promotions"
    if _contains_any(combined, SOCIAL_KEYWORDS):
        return "social"
    if _contains_any(combined, WORK_KEYWORDS):
        return "work"
    if _contains_any(combined, PERSONAL_KEYWORDS):
        return "personal"
    # Default: if subject/body sounds professional default to work, else personal
    return "work"


def _assign_label(row: pd.Series) -> str:
    """
    Try CMU annotation first; fall back to keyword heuristics.
    """
    cmu = _label_from_cmu(row)
    if cmu is not None:
        return cmu
    return _label_from_keywords(row)


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `label` column with one of 6 class values:
        spam, phishing, promotions, social, personal, work

    Uses CMU annotation columns (Cat1..Cat9, level_weight) when present,
    otherwise falls back to keyword heuristics.

    Requires columns: label_raw, subject, body
    """
    if "label_raw" not in df.columns:
        raise ValueError("DataFrame must have a 'label_raw' column (values: 'spam' or 'ham').")

    has_cmu = "cat_1_level_1" in df.columns
    if has_cmu:
        n_annotated = df["cat_1_level_1"].notna().sum()
        logger.info("CMU annotation columns detected — %d annotated rows, rest use keyword fallback.", n_annotated)
    else:
        logger.info("No CMU annotation columns found — using keyword heuristics for all records.")

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
            logger.warning(
                "  Class '%s' has very few samples (%.1f%%) — consider oversampling.",
                label, pct,
            )
    print("\nClass value_counts:")
    print(counts.to_string())
    print()
