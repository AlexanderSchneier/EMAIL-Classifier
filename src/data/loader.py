"""Load raw email files from Enron, SpamAssassin, and Nazario phishing datasets into a unified DataFrame."""

import email
import email.policy
import logging
import mailbox
import os
from pathlib import Path

import pandas as pd

from config.settings import ENRON_PATH, SPAMASSASSIN_PATH, PHISHING_PATH, MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)

# ── CSV column name mappings for common Kaggle Enron datasets ──────────────────
# Maps (text_col, label_col) → label_values meaning spam
_CSV_FORMATS = [
    # venky73/spam-mails-dataset  →  columns: label, text   (label: "spam"/"ham")
    ("text",    "label",    lambda v: str(v).strip().lower() == "spam"),
    # ozlerhakan/spam-or-not-spam →  columns: text, spam    (spam: 1/0)
    ("text",    "spam",     lambda v: int(v) == 1),
    # other common variants
    ("message", "label",    lambda v: str(v).strip().lower() == "spam"),
    ("email",   "label",    lambda v: str(v).strip().lower() == "spam"),
    ("body",    "label",    lambda v: str(v).strip().lower() == "spam"),
    ("text",    "category", lambda v: str(v).strip().lower() == "spam"),
    ("message", "category", lambda v: str(v).strip().lower() == "spam"),
]


def _parse_email_string(raw_text: str) -> dict:
    """Parse a raw email string (with headers) using the stdlib email module."""
    try:
        msg = email.message_from_string(raw_text, policy=email.policy.compat32)
    except Exception:
        return {"subject": "", "sender": "", "body": raw_text[:MAX_TEXT_LENGTH]}

    subject = msg.get("Subject", "") or ""
    sender = msg.get("From", "") or ""

    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    body_parts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body_parts.append(payload.decode(charset, errors="replace"))
        except Exception:
            body_parts.append(str(msg.get_payload() or ""))

    body = " ".join(body_parts)[:MAX_TEXT_LENGTH]
    return {"subject": subject, "sender": sender, "body": body}


def _load_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a Kaggle-style Enron CSV into the unified DataFrame schema.
    Auto-detects the text and label column names.

    Supported formats:
      - Labeled:   text/label, text/spam, message/label, etc.
      - Raw corpus: file/message  (wcukierski/enron-email-dataset)
                    All treated as ham — combine with SpamAssassin for spam classes.
    """
    df_raw = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]
    cols = set(df_raw.columns)

    # ── Labeled formats ────────────────────────────────────────────────────────
    for text_col, label_col, is_spam_fn in _CSV_FORMATS:
        if text_col in cols and label_col in cols:
            logger.info("CSV format detected: text='%s', label='%s'", text_col, label_col)
            records = []
            for _, row in df_raw.iterrows():
                body = str(row[text_col])[:MAX_TEXT_LENGTH]
                label_raw = "spam" if is_spam_fn(row[label_col]) else "ham"
                records.append({
                    "subject": "",
                    "sender": "",
                    "body": body,
                    "raw_text": body,
                    "label_raw": label_raw,
                    "source": "enron",
                })
            return pd.DataFrame(records)

    # ── Raw Enron corpus (wcukierski/enron-email-dataset): file + message ──────
    if "file" in cols and "message" in cols:
        logger.info(
            "Detected raw Enron corpus format (file + message). "
            "All emails will be labelled 'ham'. "
            "Add SpamAssassin data to get spam/phishing classes."
        )
        records = []
        for _, row in df_raw.iterrows():
            parsed = _parse_email_string(str(row["message"]))
            raw_text = f"{parsed['subject']} {parsed['body']}".strip()
            records.append({
                "subject": parsed["subject"],
                "sender":  parsed["sender"],
                "body":    parsed["body"],
                "raw_text": raw_text[:MAX_TEXT_LENGTH],
                "label_raw": "ham",
                "source": "enron",
            })
        return pd.DataFrame(records)

    raise ValueError(
        f"Could not detect columns in {filepath.name}.\n"
        f"Found columns: {list(df_raw.columns)}\n"
        f"Expected a labeled format {[(t, l) for t, l, _ in _CSV_FORMATS]} "
        f"or the raw Enron corpus format ('file', 'message')."
    )


def _parse_email_file(filepath: Path) -> dict:
    """Parse a single raw email file using the stdlib email module."""
    try:
        raw = filepath.read_bytes()
        msg = email.message_from_bytes(raw, policy=email.policy.compat32)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", filepath, e)
        return None

    subject = msg.get("Subject", "") or ""
    sender = msg.get("From", "") or ""

    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    body_parts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body_parts.append(payload.decode(charset, errors="replace"))
        except Exception:
            body_parts.append(str(msg.get_payload()))

    body = " ".join(body_parts)[:MAX_TEXT_LENGTH]
    raw_text = f"{subject} {body}".strip()

    return {
        "subject": subject,
        "sender": sender,
        "body": body,
        "raw_text": raw_text,
    }


def _load_brianray_chunks(folder: Path) -> pd.DataFrame:
    """
    Load the brianray/enron-email-dataset (CMU annotated version).

    Reads all chunk CSVs from the data/ subfolder. Passes annotation columns
    (cat_i_level_1, cat_i_level_2, cat_i_weight) through to the labeler.
    Unlabeled rows (labeled=False) will fall back to keyword heuristics.
    """
    import glob as _glob
    chunk_files = sorted(_glob.glob(str(folder / "data" / "*chunk*.csv")))
    if not chunk_files:
        # Fall back to the full file if chunks aren't present
        full = list((folder / "data").glob("*.csv"))
        chunk_files = [str(f) for f in full if "Zone" not in str(f)]

    if not chunk_files:
        logger.warning("No CSV files found in %s", folder / "data")
        return pd.DataFrame()

    logger.info("Loading brianray annotated Enron dataset (%d chunk files)...", len(chunk_files))

    # Annotation column names in this dataset
    cat_cols = []
    for i in range(1, 13):
        cat_cols += [f"cat_{i}_level_1", f"cat_{i}_level_2", f"cat_{i}_weight"]

    frames = []
    for fpath in chunk_files:
        if "Zone.Identifier" in fpath:
            continue
        df_chunk = pd.read_csv(fpath, low_memory=False)
        records = []
        for _, row in df_chunk.iterrows():
            subject = str(row.get("subject", "") or "")
            sender  = str(row.get("from", "") or "")
            body    = str(row.get("content", "") or "")[:MAX_TEXT_LENGTH]
            raw_text = f"{subject} {body}".strip()

            rec = {
                "subject":   subject,
                "sender":    sender,
                "body":      body,
                "raw_text":  raw_text,
                "label_raw": "ham",   # entire Enron corpus is legitimate email
                "source":    "enron_annotated",
            }
            # Pass annotation columns through so the labeler can use them
            for col in cat_cols:
                rec[col] = row.get(col, None)

            records.append(rec)
        frames.append(pd.DataFrame(records))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d emails from brianray annotated Enron dataset", len(combined))

    # Keep only rows with at least one CMU annotation
    if "cat_1_level_1" in combined.columns:
        annotated_mask = combined["cat_1_level_1"].notna()
        n_before = len(combined)
        combined = combined[annotated_mask].reset_index(drop=True)
        logger.info(
            "Filtered to CMU-annotated rows only: %d → %d emails",
            n_before, len(combined),
        )

    return combined


def load_enron(path: Path = ENRON_PATH) -> pd.DataFrame:
    """
    Load the Enron email dataset.

    Priority order:
      1. brianray annotated dataset (subfolder brianray-enron-email-dataset/)
      2. Plain CSV file dropped into data/raw/enron/
      3. Directory layout with spam/ and ham/ folders of .txt files
    """
    records = []
    path = Path(path)
    if not path.exists():
        logger.warning("Enron path does not exist: %s", path)
        return pd.DataFrame()

    # ── Brianray annotated dataset (highest priority — has CMU labels) ─────────
    brianray_dir = path / "brianray-enron-email-dataset"
    if brianray_dir.exists():
        logger.info("Found brianray annotated Enron dataset — using CMU annotations.")
        return _load_brianray_chunks(brianray_dir)

    # ── Plain CSV mode ─────────────────────────────────────────────────────────
    csv_files = list(path.glob("*.csv"))
    if csv_files:
        csv_path = csv_files[0]
        if len(csv_files) > 1:
            logger.warning("Multiple CSV files found in %s — using %s", path, csv_path.name)
        logger.info("Loading Enron from CSV: %s", csv_path)
        df = _load_csv(csv_path)
        logger.info("Loaded %d emails from Enron CSV", len(df))
        return df

    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        folder = root_path.name.lower()
        if folder not in ("spam", "ham"):
            continue
        label_raw = folder  # "spam" or "ham"
        for fname in files:
            fpath = root_path / fname
            if not fpath.is_file():
                continue
            parsed = _parse_email_file(fpath)
            if parsed is None:
                continue
            parsed["label_raw"] = label_raw
            parsed["source"] = "enron"
            records.append(parsed)

    df = pd.DataFrame(records)
    logger.info("Loaded %d emails from Enron dataset", len(df))
    return df


def load_spamassassin(path: Path = SPAMASSASSIN_PATH) -> pd.DataFrame:
    """
    Load the SpamAssassin public corpus.

    Expected layout:
        spamassassin/
            spam/          ← confirmed spam
            easy_ham/      ← easy legitimate email
            hard_ham/      ← tricky legitimate email
            spam_2/        ← additional spam (optional)
            easy_ham_2/    ← additional ham (optional)
    """
    records = []
    path = Path(path)
    if not path.exists():
        logger.warning("SpamAssassin path does not exist: %s", path)
        return pd.DataFrame()

    spam_dirs = {"spam", "spam_2"}

    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        folder = root_path.name.lower()
        if not any(folder.startswith(s) for s in ("spam", "easy_ham", "hard_ham")):
            continue
        label_raw = "spam" if folder in spam_dirs or folder == "spam" else "ham"
        for fname in files:
            if fname.startswith("cmds"):
                continue
            fpath = root_path / fname
            if not fpath.is_file():
                continue
            parsed = _parse_email_file(fpath)
            if parsed is None:
                continue
            parsed["label_raw"] = label_raw
            parsed["source"] = "spamassassin"
            records.append(parsed)

    df = pd.DataFrame(records)
    logger.info("Loaded %d emails from SpamAssassin dataset", len(df))
    return df


def load_phishing(path: Path = PHISHING_PATH) -> pd.DataFrame:
    """
    Load the Nazario phishing corpus (.mbox files).

    Expected layout:
        data/raw/phishing/
            phishing0.mbox
            phishing2.mbox
            phishing3.mbox
            ...

    Every message is labeled directly as 'phishing' — no heuristics needed.
    Download with: python scripts/download_data.py --phishing
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Phishing path does not exist: %s", path)
        return pd.DataFrame()

    mbox_files = list(path.glob("*.mbox"))
    if not mbox_files:
        logger.warning("No .mbox files found in %s", path)
        return pd.DataFrame()

    records = []
    for mbox_path in sorted(mbox_files):
        logger.info("Loading phishing mbox: %s", mbox_path.name)
        try:
            mbox = mailbox.mbox(str(mbox_path))
        except Exception as e:
            logger.warning("Failed to open %s: %s", mbox_path, e)
            continue

        for msg in mbox:
            try:
                subject = msg.get("Subject", "") or ""
                sender = msg.get("From", "") or ""

                body_parts = []
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                payload = part.get_payload(decode=True)
                                charset = part.get_content_charset() or "utf-8"
                                body_parts.append(payload.decode(charset, errors="replace"))
                            except Exception:
                                pass
                else:
                    try:
                        payload = msg.get_payload(decode=True)
                        if payload:
                            charset = msg.get_content_charset() or "utf-8"
                            body_parts.append(payload.decode(charset, errors="replace"))
                    except Exception:
                        body_parts.append(str(msg.get_payload() or ""))

                body = " ".join(body_parts)[:MAX_TEXT_LENGTH]
                raw_text = f"{subject} {body}".strip()

                records.append({
                    "subject": subject,
                    "sender": sender,
                    "body": body,
                    "raw_text": raw_text,
                    "label_raw": "phishing",
                    "source": "nazario",
                })
            except Exception as e:
                logger.warning("Skipping malformed message in %s: %s", mbox_path.name, e)

    df = pd.DataFrame(records)
    logger.info("Loaded %d phishing emails from Nazario corpus", len(df))
    return df


def load_all() -> pd.DataFrame:
    """Load and concatenate all available datasets."""
    frames = []
    enron_df = load_enron()
    if not enron_df.empty:
        frames.append(enron_df)

    sa_df = load_spamassassin()
    if not sa_df.empty:
        frames.append(sa_df)

    phishing_df = load_phishing()
    if not phishing_df.empty:
        frames.append(phishing_df)

    if not frames:
        raise FileNotFoundError(
            "No data found. Place email files under data/raw/enron/, data/raw/spamassassin/, "
            "and/or data/raw/phishing/."
        )

    combined = pd.concat(frames, ignore_index=True)
    # Ensure required columns exist
    for col in ("subject", "sender", "body", "raw_text", "label_raw", "source"):
        if col not in combined.columns:
            combined[col] = ""
    combined = combined.fillna("")
    logger.info("Total emails loaded: %d", len(combined))
    return combined
