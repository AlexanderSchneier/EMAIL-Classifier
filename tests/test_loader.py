"""Tests for src/data/loader.py — uses synthetic .eml files, no real dataset required."""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import _parse_email_file, load_enron, load_spamassassin


RAW_EMAIL = b"""\
From: alice@example.com
To: bob@example.com
Subject: Hello world
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

This is the email body.
"""

MULTIPART_EMAIL = b"""\
From: sender@test.com
To: recv@test.com
Subject: Multipart
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="boundary42"

--boundary42
Content-Type: text/plain; charset=utf-8

Plain text part.
--boundary42
Content-Type: application/octet-stream

<binary data>
--boundary42--
"""


def _write_email(directory: Path, filename: str, content: bytes):
    filepath = directory / filename
    filepath.write_bytes(content)
    return filepath


def test_parse_email_file_simple():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = _write_email(Path(tmpdir), "test.txt", RAW_EMAIL)
        result = _parse_email_file(fpath)
    assert result is not None
    assert result["subject"] == "Hello world"
    assert "alice@example.com" in result["sender"]
    assert "email body" in result["body"]
    assert "Hello world" in result["raw_text"]


def test_parse_email_file_multipart():
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = _write_email(Path(tmpdir), "multi.txt", MULTIPART_EMAIL)
        result = _parse_email_file(fpath)
    assert result is not None
    assert "Plain text part" in result["body"]


def test_parse_email_file_missing(tmp_path):
    result = _parse_email_file(tmp_path / "nonexistent.txt")
    assert result is None


def test_load_enron_empty_path(tmp_path):
    df = load_enron(path=tmp_path / "nonexistent")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_enron_with_files(tmp_path):
    spam_dir = tmp_path / "spam"
    ham_dir = tmp_path / "ham"
    spam_dir.mkdir()
    ham_dir.mkdir()

    _write_email(spam_dir, "s1.txt", RAW_EMAIL)
    _write_email(spam_dir, "s2.txt", RAW_EMAIL)
    _write_email(ham_dir, "h1.txt", RAW_EMAIL)

    df = load_enron(path=tmp_path)
    assert len(df) == 3
    assert set(df["label_raw"].unique()) == {"spam", "ham"}
    assert (df["label_raw"] == "spam").sum() == 2
    assert (df["label_raw"] == "ham").sum() == 1
    assert all(col in df.columns for col in ["subject", "sender", "body", "raw_text", "source"])


def test_load_spamassassin_empty_path(tmp_path):
    df = load_spamassassin(path=tmp_path / "nonexistent")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_spamassassin_with_files(tmp_path):
    spam_dir = tmp_path / "spam"
    ham_dir = tmp_path / "easy_ham"
    spam_dir.mkdir()
    ham_dir.mkdir()

    _write_email(spam_dir, "s1.txt", RAW_EMAIL)
    _write_email(ham_dir, "h1.txt", RAW_EMAIL)

    df = load_spamassassin(path=tmp_path)
    assert len(df) == 2
    assert set(df["label_raw"].unique()) == {"spam", "ham"}
    assert df["source"].iloc[0] == "spamassassin"
