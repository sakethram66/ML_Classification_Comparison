"""Train and compare four classifiers using the Part 1 prepared data."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from evaluation import (
    plot_combined_confusion_matrices,
    plot_comparison_chart,
    plot_confusion_matrices,
    save_metrics_csv,
    evaluate_model,
)
from preprocessing import PROJECT_DIR, load_and_preprocess_data


OUTPUT_DIR = PROJECT_DIR / "outputs"
GRAPH_DIR = PROJECT_DIR / "graphs"
METRICS_PATH = OUTPUT_DIR / "comparison_results.csv"
COMPARISON_PLOT_PATH = GRAPH_DIR / "model_comparison.png"
COMBINED_CONFUSION_PATH = GRAPH_DIR / "confusion_matrices.png"


def create_models() -> dict[str, Any]:
    """Create the four requested classifiers with reproducible parameters."""
    return {
        # A larger forest reduces variance; all randomness is fixed for reproducibility.
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        # RBF SVM uses the training-fitted standardized features from Part 1.
        "Support Vector Machine (SVM)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=42,
        ),
        # Gaussian Naive Bayes is deterministic and uses scikit-learn defaults.
        "Gaussian Naive Bayes": GaussianNB(),
        # Standard gradient boosting settings with fixed estimator count and seed.
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }


def _print_results(results: dict[str, dict[str, Any]]) -> None:
    """Print comparison metrics, detailed class reports, and measured highlights."""
    rows = [
        {
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision (malignant)": metrics["precision"],
            "Recall (malignant)": metrics["recall"],
            "F1 (malignant)": metrics["f1_score"],
        }
        for model_name, metrics in results.items()
    ]
    table = pd.DataFrame(rows).set_index("Model")
    print("\nCOMPARISON METRICS")
    print("Precision, recall, and F1-score treat malignant (label 0) as the positive class.")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))

    print("\nCLASSIFICATION REPORTS (both classes)")
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(metrics["classification_report"])

    best_model = max(results, key=lambda name: results[name]["f1_score"])
    best_f1 = results[best_model]["f1_score"]
    best_recall_model = max(results, key=lambda name: results[name]["recall"])
    best_recall = results[best_recall_model]["recall"]
    print(
        "\nBest model by malignant-class F1-score "
        f"(the comparison's primary selection metric): {best_model} ({best_f1:.4f})"
    )
    print(
        "Highest malignant-class recall: "
        f"{best_recall_model} ({best_recall:.4f})"
    )
    print(
        "Medical-data note: these held-out results are for educational model comparison, "
        "not clinical diagnosis or deployment."
    )


def run_experiment() -> dict[str, dict[str, Any]]:
    """Prepare shared Part 1 splits, train all models, evaluate, and save outputs."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)

        # One preprocessing run provides identical train/test rows to every classifier.
        data = load_and_preprocess_data()
        models = create_models()
        results: dict[str, dict[str, Any]] = {}

        for model_name, model in models.items():
            if model_name == "Support Vector Machine (SVM)":
                X_train = data["X_train_scaled"]
                X_test = data["X_test_scaled"]
            else:
                X_train = data["X_train"]
                X_test = data["X_test"]

            try:
                model.fit(X_train, data["y_train"])
                predictions = model.predict(X_test)
            except (ValueError, TypeError, RuntimeError) as exc:
                raise RuntimeError(f"{model_name} training or prediction failed: {exc}") from exc

            results[model_name] = evaluate_model(data["y_test"], predictions)
            print(f"\nTrained and evaluated: {model_name}")

        csv_path = save_metrics_csv(results, METRICS_PATH)
        confusion_paths = plot_confusion_matrices(results, OUTPUT_DIR)
        comparison_path = plot_comparison_chart(results, COMPARISON_PLOT_PATH)
        combined_path = plot_combined_confusion_matrices(results, COMBINED_CONFUSION_PATH)

        _print_results(results)
        print("\nGenerated evaluation files:")
        print(f"- {csv_path}")
        for path in confusion_paths:
            print(f"- {path}")
        print(f"- {comparison_path}")
        print(f"- {combined_path}")
        return results
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Model comparison failed: {exc}") from exc


if __name__ == "__main__":
    try:
        run_experiment()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
