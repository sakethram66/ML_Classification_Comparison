"""Reusable evaluation and plotting helpers for classification experiments."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


CLASS_LABELS = (0, 1)
CLASS_NAMES = ("malignant", "benign")
POSITIVE_LABEL = 0
METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "benign_precision",
    "benign_recall",
    "benign_f1_score",
)


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Calculate common metrics, malignant-positive metrics, and both-class report."""
    try:
        malignant_scores = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[POSITIVE_LABEL],
            average="binary",
            pos_label=POSITIVE_LABEL,
            zero_division=0,
        )
        per_class_scores = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average=None,
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(malignant_scores[0]),
            "recall": float(malignant_scores[1]),
            "f1_score": float(malignant_scores[2]),
            "benign_precision": float(per_class_scores[0][1]),
            "benign_recall": float(per_class_scores[1][1]),
            "benign_f1_score": float(per_class_scores[2][1]),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=CLASS_LABELS,
                target_names=CLASS_NAMES,
                zero_division=0,
                digits=4,
            ),
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=CLASS_LABELS
            ),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not evaluate model predictions: {exc}") from exc


def save_metrics_csv(
    results: dict[str, dict[str, Any]], output_path: str | Path
) -> Path:
    """Save comparable metrics for every model to a CSV file."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "model": model_name,
                "positive_class": CLASS_NAMES[0],
                **{column: metrics[column] for column in METRIC_COLUMNS},
            }
            for model_name, metrics in results.items()
        ]
        pd.DataFrame(
            rows, columns=["model", "positive_class", *METRIC_COLUMNS]
        ).to_csv(path, index=False, float_format="%.6f")
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Could not save comparison metrics to '{path}': {exc}") from exc
    return path


def plot_confusion_matrices(
    results: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Save an individual, labeled confusion-matrix plot for each model."""
    directory = Path(output_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Could not create confusion-matrix directory '{directory}': {exc}") from exc

    saved: list[Path] = []
    for model_name, metrics in results.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            sns.heatmap(
                metrics["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=ax,
            )
            ax.set_title(f"{model_name} — Confusion Matrix")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            fig.tight_layout()
            path = directory / f"{_filename(model_name)}_confusion_matrix.png"
            _save_figure(fig, path, f"{model_name} confusion matrix")
            saved.append(path)
        except Exception as exc:
            plt.close(fig)
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Could not create {model_name} confusion matrix: {exc}") from exc
    return saved


def plot_comparison_chart(
    results: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Create a grouped chart of accuracy and malignant-positive metrics."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics_frame = pd.DataFrame(
            {
                model_name: {
                    "Accuracy": values["accuracy"],
                    "Precision (malignant)": values["precision"],
                    "Recall (malignant)": values["recall"],
                    "F1-score (malignant)": values["f1_score"],
                }
                for model_name, values in results.items()
            }
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            metrics_frame.plot(kind="bar", ax=ax, width=0.82)
            ax.set_title("Classifier Performance Comparison")
            ax.set_xlabel("Evaluation metric")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.05)
            ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=0)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            fig.tight_layout()
            _save_figure(fig, path, "model comparison")
        except Exception:
            plt.close(fig)
            raise
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Could not create comparison chart at '{path}': {exc}") from exc
    return path


def plot_combined_confusion_matrices(
    results: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save one figure containing the confusion matrices for all evaluated models."""
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        model_items = list(results.items())
        if not model_items:
            raise ValueError("At least one model result is required.")
        fig, axes = plt.subplots(
            1,
            len(model_items),
            figsize=(5 * len(model_items), 4.8),
            squeeze=False,
        )
        try:
            for ax, (model_name, metrics) in zip(axes.flat, model_items):
                sns.heatmap(
                    metrics["confusion_matrix"],
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES,
                    ax=ax,
                )
                ax.set_title(model_name)
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
            fig.suptitle("Confusion Matrices — All Classifiers", y=1.03)
            fig.tight_layout()
            _save_figure(fig, path, "combined confusion matrices")
        except Exception:
            plt.close(fig)
            raise
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Could not create combined confusion matrices at '{path}': {exc}") from exc
    return path


def _filename(model_name: str) -> str:
    """Convert a displayed model name into a stable, filesystem-safe filename."""
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def _save_figure(fig: plt.Figure, path: Path, description: str) -> None:
    """Save a figure at 300 DPI and close it even if saving fails."""
    try:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    except Exception as exc:
        raise RuntimeError(f"Could not save {description} to '{path}': {exc}") from exc
    finally:
        plt.close(fig)
