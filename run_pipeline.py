"""
End-to-end email classification pipeline.

Usage:
    python run_pipeline.py [--dataset enron|spamassassin|both] [--models all|nb|lr|svm|rf|nn]

Example:
    python run_pipeline.py --dataset both --models all
"""

import argparse
import logging
import sys
from pathlib import Path

import scipy.sparse as sp
from sklearn.model_selection import train_test_split

# ── Make src/ importable ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    LABELS,
)
from src.data.loader import load_enron, load_spamassassin, load_all
from src.data.preprocessor import preprocess
from src.data.labeler import apply_labels
from src.features.tfidf_features import TfidfFeatures
from src.features.ngram_features import CharNgramFeatures
from src.features.metadata_features import MetadataFeatures
from src.models.naive_bayes import NaiveBayesClassifier
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.svm import SVMClassifier
from src.models.random_forest import RandomForestEmailClassifier
from src.models.neural_network import NeuralNetworkClassifier
from src.evaluation.metrics import compute_metrics, save_metrics_table, compare_models
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_label_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Email multi-class classifier pipeline")
    p.add_argument(
        "--dataset",
        choices=["enron", "spamassassin", "both"],
        default="both",
        help="Which dataset(s) to use (default: both)",
    )
    p.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys or 'all'. Options: nb,lr,svm,rf,nn",
    )
    p.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Load preprocessed CSV from data/processed/ instead of re-parsing raw files",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Randomly sample N emails before preprocessing (useful for quick testing)",
    )
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(dataset: str, skip_preprocessing: bool, max_samples: int = None):
    processed_csv = PROCESSED_DIR / "emails_processed.csv"

    if skip_preprocessing and processed_csv.exists():
        import pandas as pd
        logger.info("Loading preprocessed data from %s", processed_csv)
        return pd.read_csv(processed_csv)

    logger.info("Loading raw email data (dataset=%s)...", dataset)
    if dataset == "enron":
        df = load_enron()
    elif dataset == "spamassassin":
        df = load_spamassassin()
    else:
        df = load_all()

    if df.empty:
        raise SystemExit(
            "ERROR: No emails loaded. "
            "Make sure email files are in data/raw/enron/ and/or data/raw/spamassassin/."
        )

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=RANDOM_SEED).reset_index(drop=True)
        logger.info("Sampled %d emails (--max-samples)", max_samples)

    logger.info("Preprocessing text...")
    df = preprocess(df)

    logger.info("Applying multi-class labels...")
    df = apply_labels(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_csv, index=False)
    logger.info("Processed data saved to %s", processed_csv)
    return df


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(df_train, df_test):
    """Fit all feature extractors on train, transform both. Returns sparse matrices."""

    # TF-IDF word n-grams
    tfidf = TfidfFeatures()
    X_train_tfidf = tfidf.fit_transform(df_train["clean_text"])
    X_test_tfidf = tfidf.transform(df_test["clean_text"])

    # Character n-grams
    char_ng = CharNgramFeatures()
    X_train_char = char_ng.fit_transform(df_train["clean_text"])
    X_test_char = char_ng.transform(df_test["clean_text"])

    # Metadata (sender, subject stats)
    meta = MetadataFeatures()
    X_train_meta = meta.fit_transform(df_train)
    X_test_meta = meta.transform(df_test)

    # Stack all feature sets
    X_train_full = sp.hstack([X_train_tfidf, X_train_char, X_train_meta], format="csr")
    X_test_full = sp.hstack([X_test_tfidf, X_test_char, X_test_meta], format="csr")

    logger.info(
        "Feature matrix: train=%s, test=%s", X_train_full.shape, X_test_full.shape
    )
    return X_train_full, X_test_full, X_train_tfidf, X_test_tfidf, tfidf


# ── Model registry ────────────────────────────────────────────────────────────

ALL_MODEL_KEYS = ["nb", "lr", "svm", "rf", "nn"]


def build_models(model_keys: list):
    registry = {
        "nb":  ("Naive Bayes",          NaiveBayesClassifier()),
        "lr":  ("Logistic Regression",  LogisticRegressionClassifier()),
        "svm": ("Linear SVM",           SVMClassifier()),
        "rf":  ("Random Forest",        RandomForestEmailClassifier()),
        "nn":  ("Neural Network (MLP)", NeuralNetworkClassifier()),
    }
    return [(key, *registry[key]) for key in model_keys if key in registry]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve model keys
    if args.models.strip().lower() == "all":
        model_keys = ALL_MODEL_KEYS
    else:
        model_keys = [k.strip() for k in args.models.split(",")]

    # ── Load & split ──────────────────────────────────────────────────────────
    df = load_data(args.dataset, args.skip_preprocessing, args.max_samples)

    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    logger.info("Train: %d  Test: %d", len(df_train), len(df_test))

    y_train = df_train["label"].values
    y_test = df_test["label"].values

    # ── Label distribution plot ───────────────────────────────────────────────
    plot_label_distribution(df)

    # ── Feature extraction ────────────────────────────────────────────────────
    X_train_full, X_test_full, X_train_tfidf, X_test_tfidf, tfidf = extract_features(
        df_train, df_test
    )

    # ── Train & evaluate each model ───────────────────────────────────────────
    models = build_models(model_keys)

    for key, name, clf in models:
        logger.info("=" * 60)
        logger.info("Training: %s", name)

        # Naive Bayes only accepts non-negative TF-IDF features
        if key == "nb":
            X_tr, X_te = X_train_tfidf, X_test_tfidf
        else:
            X_tr, X_te = X_train_full, X_test_full

        clf.fit(X_tr, y_train)

        # Expose feature names to logistic regression for interpretability
        if key == "lr" and hasattr(clf, "set_feature_names"):
            clf.set_feature_names(tfidf.get_feature_names())
            top = clf.get_top_features_per_class(n=5)
            for cls, feats in top.items():
                words = [f[0] for f in feats]
                logger.info("  Top features [%s]: %s", cls, words)

        y_pred = clf.predict(X_te)
        metrics = compute_metrics(y_test, y_pred, label_names=LABELS)
        save_metrics_table(metrics, model_name=name)
        plot_confusion_matrix(y_test, y_pred, label_names=LABELS, model_name=name)

    # ── Final comparison ──────────────────────────────────────────────────────
    summary = compare_models()
    if not summary.empty:
        logger.info("=" * 60)
        logger.info("Model comparison (sorted by macro F1):")
        logger.info("\n%s", summary.to_string(index=False))
        plot_model_comparison(summary)

    logger.info("Done. Results saved to outputs/")


if __name__ == "__main__":
    main()
