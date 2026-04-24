"""Load raw email files from Enron (CMU annotated) and Nazario phishing datasets into a unified DataFrame."""

import email
import email.policy
import logging
import mailbox
from pathlib import Path

import pandas as pd

from config.settings import ENRON_PATH, PHISHING_PATH, MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)


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


def _load_brianray_chunks(folder: Path) -> pd.DataFrame:
    """
    Load the brianray/enron-email-dataset (CMU annotated version).

    Reads all chunk CSVs from the data/ subfolder. Passes annotation columns
    (cat_i_level_1, cat_i_level_2, cat_i_weight) through to the labeler.
    """
    import glob as _glob
    chunk_files = sorted(_glob.glob(str(folder / "data" / "*chunk*.csv")))
    if not chunk_files:
        full = list((folder / "data").glob("*.csv"))
        chunk_files = [str(f) for f in full if "Zone" not in str(f)]

    if not chunk_files:
        logger.warning("No CSV files found in %s", folder / "data")
        return pd.DataFrame()

    logger.info("Loading brianray annotated Enron dataset (%d chunk files)...", len(chunk_files))

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
                "label_raw": "ham",
                "source":    "enron_annotated",
            }
            for col in cat_cols:
                rec[col] = row.get(col, None)

            records.append(rec)
        frames.append(pd.DataFrame(records))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d emails from brianray annotated Enron dataset", len(combined))

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
    """Load the CMU-annotated brianray Enron dataset."""
    path = Path(path)
    if not path.exists():
        logger.warning("Enron path does not exist: %s", path)
        return pd.DataFrame()

    brianray_dir = path / "brianray-enron-email-dataset"
    if not brianray_dir.exists():
        logger.warning("brianray-enron-email-dataset not found in %s", path)
        return pd.DataFrame()

    return _load_brianray_chunks(brianray_dir)


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
    """Load and concatenate Enron (CMU annotated) and Nazario phishing datasets."""
    frames = []
    enron_df = load_enron()
    if not enron_df.empty:
        frames.append(enron_df)

    phishing_df = load_phishing()
    if not phishing_df.empty:
        frames.append(phishing_df)

    if not frames:
        raise FileNotFoundError(
            "No data found. Place the brianray Enron dataset under data/raw/enron/ "
            "and/or Nazario .mbox files under data/raw/phishing/."
        )

    combined = pd.concat(frames, ignore_index=True)
    for col in ("subject", "sender", "body", "raw_text", "label_raw", "source"):
        if col not in combined.columns:
            combined[col] = ""
    combined = combined.fillna("")
    logger.info("Total emails loaded: %d", len(combined))
    return combined
