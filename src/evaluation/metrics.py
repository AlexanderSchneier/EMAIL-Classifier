"""Compute and persist evaluation metrics."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from config.settings import LABELS, RESULTS_DIR

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, label_names=None) -> dict:
    """
    Return a dict with overall accuracy and per-class precision/recall/F1.
    """
    if label_names is None:
        label_names = LABELS

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=label_names, output_dict=True, zero_division=0
    )
    logger.info("Accuracy: %.4f", acc)
    logger.info(
        "Macro F1: %.4f", report.get("macro avg", {}).get("f1-score", 0)
    )
    return {"accuracy": acc, "report": report}


def save_metrics_table(metrics: dict, model_name: str, output_path: Path = None) -> Path:
    """
    Append one row per class (plus averages) to a cumulative CSV comparison file.
    """
    if output_path is None:
        output_path = RESULTS_DIR / "model_comparison.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = metrics["report"]
    rows = []
    for class_label, vals in report.items():
        if class_label in ("accuracy",):
            continue
        if isinstance(vals, dict):
            rows.append(
                {
                    "model": model_name,
                    "class": class_label,
                    "precision": vals.get("precision", 0),
                    "recall": vals.get("recall", 0),
                    "f1_score": vals.get("f1-score", 0),
                    "support": vals.get("support", 0),
                }
            )

    # prepend overall accuracy row
    rows.insert(
        0,
        {
            "model": model_name,
            "class": "overall_accuracy",
            "precision": metrics["accuracy"],
            "recall": metrics["accuracy"],
            "f1_score": metrics["accuracy"],
            "support": sum(v.get("support", 0) for k, v in report.items() if isinstance(v, dict)),
        },
    )

    df_new = pd.DataFrame(rows)

    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        # Remove old rows for this model to allow re-running
        df_existing = df_existing[df_existing["model"] != model_name]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(output_path, index=False)
    logger.info("Metrics saved to %s", output_path)
    return output_path


def compare_models(results_path: Path = None) -> pd.DataFrame:
    """
    Load the comparison CSV and return a summary DataFrame sorted by macro F1.
    """
    if results_path is None:
        results_path = RESULTS_DIR / "model_comparison.csv"
    results_path = Path(results_path)

    if not results_path.exists():
        logger.warning("No results file found at %s", results_path)
        return pd.DataFrame()

    df = pd.read_csv(results_path)
    macro_rows = df[df["class"] == "macro avg"].copy()
    macro_rows = macro_rows.sort_values("f1_score", ascending=False)
    return macro_rows[["model", "precision", "recall", "f1_score"]].reset_index(drop=True)
