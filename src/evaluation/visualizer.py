"""Plotting utilities for model evaluation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config.settings import FIGURES_DIR, LABELS

logger = logging.getLogger(__name__)

# Use non-interactive backend so plots save without a display
plt.switch_backend("Agg")


def plot_confusion_matrix(
    y_true,
    y_pred,
    label_names=None,
    model_name: str = "model",
    save_path: Path = None,
) -> Path:
    """Save a normalised confusion matrix heatmap."""
    if label_names is None:
        label_names = LABELS

    if save_path is None:
        save_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Only include labels that actually appear
    present = sorted(set(y_true) | set(y_pred))
    present_names = [l for l in label_names if l in present]

    cm = confusion_matrix(y_true, y_pred, labels=present_names, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=present_names,
        yticklabels=present_names,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)
    return save_path


def plot_model_comparison(summary_df: pd.DataFrame, save_path: Path = None) -> Path:
    """Grouped bar chart comparing macro metrics across all models."""
    if summary_df.empty:
        logger.warning("No data to plot for model comparison.")
        return None

    if save_path is None:
        save_path = FIGURES_DIR / "model_comparison.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["precision", "recall", "f1_score"]
    x = np.arange(len(summary_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, summary_df[metric], width, label=metric.replace("_", " ").title())

    ax.set_xticks(x + width)
    ax.set_xticklabels(summary_df["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Macro Averages")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Model comparison chart saved to %s", save_path)
    return save_path


def plot_label_distribution(df: pd.DataFrame, save_path: Path = None) -> Path:
    """Countplot of class label distribution."""
    if save_path is None:
        save_path = FIGURES_DIR / "label_distribution.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    order = df["label"].value_counts().index.tolist()
    sns.countplot(data=df, x="label", order=order, ax=ax, palette="Set2")
    ax.set_title("Email Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Label distribution chart saved to %s", save_path)
    return save_path
