
"""
Download email datasets for the classifier pipeline.

SpamAssassin: downloaded automatically from Apache's public corpus (no account needed).
Enron:        downloaded via the Kaggle API (requires a free Kaggle account).

Usage:
    # Download only SpamAssassin (no account needed)
    python scripts/download_data.py --spamassassin

    # Download only Enron (requires kaggle credentials)
    python scripts/download_data.py --enron

    # Download both
    python scripts/download_data.py --all

Kaggle setup (one-time, only needed for Enron):
    1. Create a free account at https://www.kaggle.com
    2. Go to https://www.kaggle.com/settings → API → "Create New Token"
    3. This downloads kaggle.json — place it at ~/.kaggle/kaggle.json
    4. Run: chmod 600 ~/.kaggle/kaggle.json
    5. Run: pip install kaggle
"""

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import ENRON_PATH, SPAMASSASSIN_PATH

# ── SpamAssassin corpus files (Apache public corpus, no auth required) ─────────
SPAMASSASSIN_FILES = [
    ("20030228_spam.tar.bz2",     "spam"),
    ("20030228_spam_2.tar.bz2",   "spam_2"),
    ("20030228_easy_ham.tar.bz2", "easy_ham"),
    ("20030228_hard_ham.tar.bz2", "hard_ham"),
    ("20030228_easy_ham_2.tar.bz2", "easy_ham_2"),
]
SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"

# ── Enron dataset on Kaggle ────────────────────────────────────────────────────
# Dataset: https://www.kaggle.com/datasets/wanderfj/enron-spam
# Contains individual .txt email files organised into spam/ and ham/ folders.
ENRON_KAGGLE_DATASET = "wanderfj/enron-spam"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress(count, block_size, total_size):
    pct = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    pct = min(pct, 100)
    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct}%", end="", flush=True)


def _download_file(url: str, dest: Path):
    print(f"  Downloading {url.split('/')[-1]} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _extract_tar(archive: Path, dest_dir: Path):
    print(f"  Extracting {archive.name} ...")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(path=dest_dir)
    archive.unlink()


# ── SpamAssassin ──────────────────────────────────────────────────────────────

def download_spamassassin():
    print("\n=== SpamAssassin Public Corpus ===")
    SPAMASSASSIN_PATH.mkdir(parents=True, exist_ok=True)

    for filename, folder in SPAMASSASSIN_FILES:
        dest_folder = SPAMASSASSIN_PATH / folder
        if dest_folder.exists() and any(dest_folder.iterdir()):
            print(f"  {folder}/ already exists — skipping.")
            continue

        url = SPAMASSASSIN_BASE_URL + filename
        archive_path = SPAMASSASSIN_PATH / filename
        try:
            _download_file(url, archive_path)
            _extract_tar(archive_path, SPAMASSASSIN_PATH)

            # The tarball extracts to a folder named e.g. "spam" or "easy_ham"
            # Rename to match the expected folder name if needed
            extracted = SPAMASSASSIN_PATH / folder
            if not extracted.exists():
                # Try to find what was actually extracted
                candidates = [
                    d for d in SPAMASSASSIN_PATH.iterdir()
                    if d.is_dir() and folder.split("_")[0] in d.name
                ]
                if candidates:
                    candidates[0].rename(extracted)

            count = len(list(extracted.glob("*"))) if extracted.exists() else 0
            print(f"  {folder}/ ready — {count} files.")
        except Exception as e:
            print(f"  ERROR downloading {filename}: {e}")

    print("SpamAssassin download complete.")
    print(f"Location: {SPAMASSASSIN_PATH}")


# ── Enron ─────────────────────────────────────────────────────────────────────

def download_enron():
    print("\n=== Enron-Spam Dataset (via Kaggle) ===")

    # Check kaggle is installed
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("       Run: pip install kaggle")
        sys.exit(1)

    # Check credentials exist
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json")
        print()
        print("To set up Kaggle credentials:")
        print("  1. Create a free account at https://www.kaggle.com")
        print("  2. Go to https://www.kaggle.com/settings → API → 'Create New Token'")
        print("  3. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

    if ENRON_PATH.exists() and any(ENRON_PATH.rglob("*.txt")):
        count = len(list(ENRON_PATH.rglob("*.txt")))
        print(f"  Enron data already exists ({count} .txt files) — skipping.")
        return

    ENRON_PATH.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading dataset: {ENRON_KAGGLE_DATASET}")
    print(f"  Destination: {ENRON_PATH}")

    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_and_extract(
        ENRON_KAGGLE_DATASET,
        path=str(ENRON_PATH),
    )

    # The dataset extracts as multiple enron1..enron6 folders each with spam/ and ham/
    # This is already the correct format for our loader — no restructuring needed.
    count = len(list(ENRON_PATH.rglob("*.txt")))
    print(f"  Enron download complete — {count} email files.")
    print(f"  Location: {ENRON_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Download email datasets")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--spamassassin", action="store_true", help="Download SpamAssassin corpus only")
    group.add_argument("--enron", action="store_true", help="Download Enron dataset only (requires Kaggle)")
    group.add_argument("--all", action="store_true", help="Download both datasets")
    args = p.parse_args()

    if args.spamassassin or args.all:
        download_spamassassin()

    if args.enron or args.all:
        download_enron()

    print("\nAll done. You can now run the pipeline:")
    print("  python run_pipeline.py --dataset both --models all")


if __name__ == "__main__":
    main()
