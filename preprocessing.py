"""Load, explore, visualize, and prepare the breast cancer dataset for Part 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "dataset_summary.txt"
TARGET_NAMES = {0: "malignant", 1: "benign"}
REPRESENTATIVE_FEATURES = ("mean radius", "mean texture", "mean smoothness")


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the built-in dataset as a feature DataFrame and target Series."""
    try:
        dataset = load_breast_cancer(as_frame=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load the scikit-learn breast cancer dataset: {exc}") from exc

    X = dataset.data.copy()
    y = dataset.target.copy()
    y.name = "diagnosis"

    print("Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Feature names: {', '.join(X.columns)}")
    print(f"Target names: {', '.join(dataset.target_names)}")
    print(f"Target mapping: {TARGET_NAMES}")
    print("The dataset is included with scikit-learn; no external CSV download is required.")
    print("\nFirst five feature rows:")
    print(X.head().to_string())
    print("\nFirst five target values:")
    print(y.head().to_string())
    return X, y


def _quality_checks(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Measure data-quality issues without dropping rows or columns."""
    missing_features = X.isna().sum()
    missing_target = int(y.isna().sum())
    non_numeric_columns = [
        str(column) for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])
    ]
    numeric = X.select_dtypes(include="number")
    non_finite_features = pd.Series(
        (~np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))).sum(axis=0),
        index=numeric.columns,
        name="non_finite_count",
        dtype="int64",
    )

    combined = pd.concat([X, y.rename("diagnosis")], axis=1)
    duplicate_rows = int(combined.duplicated().sum())
    return {
        "missing_features": missing_features,
        "missing_target": missing_target,
        "non_finite_features": non_finite_features,
        "non_numeric_columns": non_numeric_columns,
        "duplicate_rows": duplicate_rows,
    }


def explore_dataset(
    X: pd.DataFrame, y: pd.Series, summary_path: str | Path = SUMMARY_PATH
) -> str:
    """Report dataset structure, statistics, and data-quality findings to a text file."""
    if len(X) != len(y):
        raise ValueError("Feature and target row counts must match for dataset exploration.")

    checks = _quality_checks(X, y)
    class_counts = y.value_counts().reindex(TARGET_NAMES, fill_value=0)
    class_percentages = y.value_counts(normalize=True).reindex(TARGET_NAMES, fill_value=0) * 100
    numeric = X.select_dtypes(include="number")
    if numeric.shape[1]:
        numeric_stats = numeric.agg(["min", "max", "mean", "std"]).T
        numeric_stats.columns = ["minimum", "maximum", "mean", "standard deviation"]
    else:
        numeric_stats = pd.DataFrame(
            columns=["minimum", "maximum", "mean", "standard deviation"]
        )

    if checks["non_numeric_columns"]:
        quality_status = "Issues found: non-numeric feature columns require attention."
    elif checks["missing_features"].sum() or checks["missing_target"]:
        quality_status = "Issues found: missing values require attention."
    elif checks["non_finite_features"].sum():
        quality_status = "Issues found: non-finite numerical values require attention."
    elif checks["duplicate_rows"]:
        quality_status = "Issues found: duplicate rows require attention."
    else:
        quality_status = "Passed: no missing, duplicate, non-finite, or non-numeric feature data."

    report = "\n".join(
        [
            "BREAST CANCER WISCONSIN (DIAGNOSTIC) — DATASET SUMMARY",
            "=" * 66,
            "Source: sklearn.datasets.load_breast_cancer() (built into scikit-learn)",
            "External data download required: No",
            f"Dataset shape: {X.shape[0]} rows x {X.shape[1]} features",
            f"Target shape: {y.shape}",
            "",
            "FEATURE DATA TYPES",
            X.dtypes.to_string(),
            "",
            "DESCRIPTIVE STATISTICS (Pandas describe())",
            X.describe().to_string(),
            "",
            "MISSING VALUES PER FEATURE",
            checks["missing_features"].to_string(),
            f"Missing target values: {checks['missing_target']}",
            "",
            "NON-NUMERIC FEATURE COLUMNS",
            (
                ", ".join(checks["non_numeric_columns"])
                if checks["non_numeric_columns"]
                else "None"
            ),
            "",
            "NON-FINITE NUMERICAL VALUES PER FEATURE",
            (
                checks["non_finite_features"].to_string()
                if len(checks["non_finite_features"])
                else "No numerical feature columns are present."
            ),
            "",
            f"Duplicate complete rows (features and target): {checks['duplicate_rows']}",
            "",
            "TARGET CLASS COUNTS AND PERCENTAGES",
            "\n".join(
                f"{TARGET_NAMES[int(label)]} (label {int(label)}): {int(count)} "
                f"({class_percentages.loc[label]:.2f}%)"
                for label, count in class_counts.items()
            ),
            "",
            "NUMERICAL FEATURE MINIMUM, MAXIMUM, MEAN, AND STANDARD DEVIATION",
            numeric_stats.to_string(float_format=lambda value: f"{value:.6f}"),
            "",
            "DATA QUALITY",
            quality_status,
            "Records and features are not automatically removed or imputed. If a check fails, "
            "preprocessing stops with an explicit error so the issue can be reviewed.",
        ]
    )

    path = Path(summary_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not write the dataset summary to '{path}': {exc}") from exc

    print("\nDataset shape:", X.shape)
    print("\nFeature data types:")
    print(X.dtypes.to_string())
    print("\nDescriptive statistics:")
    print(X.describe().to_string())
    print("\nMissing values per feature:")
    print(checks["missing_features"].to_string())
    print(f"\nDuplicate complete rows: {checks['duplicate_rows']}")
    print("\nTarget class counts:")
    print(report.split("TARGET CLASS COUNTS AND PERCENTAGES\n", 1)[1].split("\n\n", 1)[0])
    print(f"\nData quality: {quality_status}")
    print(f"Full summary saved to: {path}")
    return report


def _save_figure(fig: plt.Figure, path: Path, description: str) -> None:
    """Save one figure at print quality and always close it."""
    try:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    except Exception as exc:
        raise RuntimeError(f"Could not save the {description} plot to '{path}': {exc}") from exc
    finally:
        plt.close(fig)


def visualize_dataset(
    X: pd.DataFrame, y: pd.Series, output_dir: str | Path = OUTPUT_DIR
) -> list[Path]:
    """Create class-count, representative-feature, and correlation plots."""
    if len(X) != len(y):
        raise ValueError("Feature and target row counts must match for visualization.")

    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Could not create the output directory '{output_path}': {exc}") from exc

    plot_data = X.copy()
    plot_data["diagnosis"] = y.map(TARGET_NAMES)
    generated: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        sns.countplot(
            data=plot_data,
            x="diagnosis",
            order=["malignant", "benign"],
            hue="diagnosis",
            hue_order=["malignant", "benign"],
            palette={"malignant": "#c44e52", "benign": "#4c72b0"},
            legend=False,
            ax=ax,
        )
        ax.set_title("Breast Cancer Diagnosis Class Distribution")
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("Number of samples")
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", padding=3)
        fig.tight_layout()
        path = output_path / "class_distribution.png"
        _save_figure(fig, path, "class distribution")
        generated.append(path)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(f"Could not create the class distribution plot: {exc}") from exc

    missing_representatives = [name for name in REPRESENTATIVE_FEATURES if name not in X.columns]
    if missing_representatives:
        raise ValueError(f"Representative features are missing from the dataset: {missing_representatives}")

    fig, axes = plt.subplots(1, len(REPRESENTATIVE_FEATURES), figsize=(15, 4.5))
    try:
        for ax, feature in zip(axes, REPRESENTATIVE_FEATURES):
            sns.histplot(
                data=plot_data,
                x=feature,
                hue="diagnosis",
                hue_order=["malignant", "benign"],
                bins=25,
                element="step",
                stat="density",
                common_norm=False,
                alpha=0.35,
                ax=ax,
            )
            ax.set_title(feature.replace("_", " ").title())
            ax.set_xlabel(feature.replace("_", " ").title())
            ax.set_ylabel("Density")
        fig.suptitle("Distributions of Representative Features by Diagnosis", y=1.04)
        fig.tight_layout()
        path = output_path / "feature_distributions.png"
        _save_figure(fig, path, "feature distributions")
        generated.append(path)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(f"Could not create the feature distribution plots: {exc}") from exc

    fig, ax = plt.subplots(figsize=(17, 14))
    try:
        correlation = X.corr(numeric_only=True)
        if correlation.empty:
            raise ValueError("No numerical features are available to calculate correlations.")
        sns.heatmap(
            correlation,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.15,
            cbar_kws={"label": "Pearson correlation"},
            ax=ax,
        )
        ax.set_title("Correlation Heatmap of Numerical Features", pad=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
        fig.tight_layout()
        path = output_path / "correlation_heatmap.png"
        _save_figure(fig, path, "correlation heatmap")
        generated.append(path)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(f"Could not create the correlation heatmap: {exc}") from exc

    print("\nGenerated plots:")
    for path in generated:
        print(f"- {path}")
    return generated


def preprocess_data(
    X: pd.DataFrame, y: pd.Series, summary_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate data, split it, and scale using training features only."""
    if len(X) != len(y):
        raise ValueError("Feature and target row counts must match before preprocessing.")
    if not X.index.equals(y.index):
        raise ValueError("Feature and target indices must match before preprocessing.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All input features must be numerical; no columns were removed.")
    if y.isna().any():
        raise ValueError("Target contains missing values; no rows were removed.")
    if y.nunique() < 2:
        raise ValueError("At least two target classes are required for a stratified split.")

    checks = _quality_checks(X, y)
    problems: list[str] = []
    missing_features = checks["missing_features"]
    if missing_features.sum():
        affected = missing_features[missing_features > 0].to_dict()
        problems.append(f"missing feature values: {affected}")
    if checks["missing_target"]:
        problems.append(f"{checks['missing_target']} missing target values")
    if checks["non_finite_features"].sum():
        affected = checks["non_finite_features"][checks["non_finite_features"] > 0].to_dict()
        problems.append(f"non-finite feature values: {affected}")
    if checks["duplicate_rows"]:
        problems.append(f"{checks['duplicate_rows']} duplicate complete rows")
    if problems:
        raise ValueError(
            "Data-quality problems detected; no records or features were silently removed: "
            + "; ".join(problems)
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    except ValueError as exc:
        raise ValueError(f"Could not create the requested stratified train/test split: {exc}") from exc

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    train_counts = y_train.value_counts().reindex(TARGET_NAMES, fill_value=0)
    test_counts = y_test.value_counts().reindex(TARGET_NAMES, fill_value=0)
    split_report = "\n".join(
        [
            "",
            "TRAIN/TEST SPLIT AND SCALING",
            "-" * 36,
            f"Split settings: test_size=0.20, random_state=42, stratify=y",
            f"Training feature shape: {X_train.shape}",
            f"Testing feature shape: {X_test.shape}",
            "Training class counts: "
            + ", ".join(
                f"{TARGET_NAMES[int(label)]}={int(count)}"
                for label, count in train_counts.items()
            ),
            "Testing class counts: "
            + ", ".join(
                f"{TARGET_NAMES[int(label)]}={int(count)}"
                for label, count in test_counts.items()
            ),
            "StandardScaler was fitted on X_train only and then used to transform X_train and X_test.",
            "Original unscaled X_train and X_test are preserved.",
            "Target labels were not scaled.",
            "SVM should use scaled features. Random Forest, Gaussian Naive Bayes, and "
            "Gradient Boosting can initially use unscaled features.",
        ]
    )
    if summary_path is not None:
        path = Path(summary_path)
        try:
            with path.open("a", encoding="utf-8") as summary_file:
                summary_file.write(split_report + "\n")
        except OSError as exc:
            raise RuntimeError(f"Could not append preprocessing details to '{path}': {exc}") from exc

    print("\nTrain/test split:")
    print(f"X_train: {X_train.shape}; X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}; y_test: {y_test.shape}")
    print(
        "Training classes: "
        + ", ".join(
            f"{TARGET_NAMES[int(label)]}={int(count)}" for label, count in train_counts.items()
        )
    )
    print(
        "Testing classes: "
        + ", ".join(
            f"{TARGET_NAMES[int(label)]}={int(count)}" for label, count in test_counts.items()
        )
    )
    print("StandardScaler fitted on X_train only; original unscaled splits are preserved.")
    print(
        "Use scaled features for SVM. Random Forest, Gaussian Naive Bayes, and "
        "Gradient Boosting can initially use unscaled features."
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


def load_and_preprocess_data() -> dict[str, Any]:
    """Run the complete Part 1 workflow and return data ready for Part 2."""
    X, y = load_dataset()
    explore_dataset(X, y)
    visualize_dataset(X, y)
    return preprocess_data(X, y, summary_path=SUMMARY_PATH)


if __name__ == "__main__":
    try:
        load_and_preprocess_data()
        print(f"\nPart 1 completed successfully. Report: {SUMMARY_PATH}")
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Part 1 failed: {exc}") from exc
