# Comparative Performance Analysis of Machine Learning Classification Algorithms Using Scikit-learn

## Project overview

This B.Tech mini project explores the Breast Cancer Wisconsin (Diagnostic) dataset and compares four scikit-learn classification algorithms on one reproducible, stratified holdout split. It covers dataset quality checks, exploratory visualizations, leakage-safe feature scaling for SVM, model evaluation, and a final written report.

> **Educational-use disclaimer:** This is a small classroom experiment, not a clinically validated diagnostic system. Its predictions must not be used for medical decisions.

## Problem statement

Implement and compare classification algorithms on a selected dataset and identify the algorithm with the best results using appropriate evaluation metrics, with attention to detecting malignant cases.

## Objectives

- Explore and prepare the dataset without silently dropping records or features.
- Train and evaluate the four specified classifiers using the same train/test examples.
- Compare accuracy and class-specific precision, recall, and F1-score.
- Interpret confusion matrices, prioritizing malignant-class recall as an important safety-related measure.
- Document the observed results and project limitations.

## Dataset

The project loads the Breast Cancer Wisconsin (Diagnostic) dataset directly from `sklearn.datasets.load_breast_cancer()`. It contains 569 samples, 30 numerical features, and two diagnoses: **0 = malignant** and **1 = benign**. No external CSV is needed. There are 212 malignant samples (37.26%) and 357 benign samples (62.74%). Part 1 found no missing values, non-finite numerical values, or duplicate complete rows.

## Algorithms

1. **Random Forest Classifier** — 300 decision trees, `random_state=42`, parallel tree fitting.
2. **Support Vector Classifier (SVM)** — RBF kernel, `C=1.0`, `gamma="scale"`, and standardized inputs.
3. **Gaussian Naive Bayes** — Gaussian class-conditional feature likelihoods, scikit-learn defaults.
4. **Gradient Boosting Classifier** — 100 boosting stages, learning rate 0.1, tree depth 3, `random_state=42`.

## Project structure

```text
ML_Classification_Comparison/
├── preprocessing.py
├── main.py
├── evaluation.py
├── requirements.txt
├── README.md
├── project_report.docx
├── data/
│   └── .gitkeep
├── graphs/
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   └── .gitkeep
├── outputs/
│   ├── dataset_summary.txt
│   ├── comparison_results.csv
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── random_forest_confusion_matrix.png
│   ├── support_vector_machine_svm_confusion_matrix.png
│   ├── gaussian_naive_bayes_confusion_matrix.png
│   └── gradient_boosting_confusion_matrix.png
└── tests/
    ├── test_preprocessing.py
    └── test_evaluation.py
```

## Installation (Windows)

Python 3.10 or later is recommended. In PowerShell, open the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, Command Prompt can use:

```bat
.venv\Scripts\activate.bat
```

## Run the project

Run the Part 1 exploration and preprocessing independently:

```powershell
python preprocessing.py
```

Run Part 2 to prepare the shared split, train all four models, evaluate, and regenerate its output files:

```powershell
python main.py
```

Run the lightweight project tests:

```powershell
python -m unittest discover -s tests -v
```

The Word report is `project_report.docx`. `python-docx` was used to generate it; it is not needed to run Part 1, Part 2, or the tests and is therefore not an experiment runtime requirement.

## Split, scaling, and evaluation definitions

Part 1 uses `test_size=0.20`, `random_state=42`, and `stratify=y`, resulting in 455 training and 114 testing samples. `StandardScaler` is fitted only on the training features; SVM receives those scaled train/test features. Random Forest, Gaussian Naive Bayes, and Gradient Boosting receive the original unscaled features. All models use the same test rows.

For the project comparison, **malignant (label 0) is the positive class** for precision, recall, and F1-score. Accuracy is the fraction of all predictions that are correct. Positive-class precision is `TP / (TP + FP)`, recall (sensitivity) is `TP / (TP + FN)`, and F1 is `2 * precision * recall / (precision + recall)`. The report also provides benign-class metrics and both-class classification reports. Zero-division cases are handled with `zero_division=0`.

## Generated reports, results, and graphs

- `outputs/dataset_summary.txt` — dataset structure, statistics, class counts, data-quality checks, and split details.
- `outputs/comparison_results.csv` — measured metrics for all four models. Precision/recall/F1 columns refer to malignant; `benign_*` columns report benign metrics.
- `outputs/*_confusion_matrix.png` — individual confusion matrices, with actual classes as rows and predicted classes as columns.
- `graphs/model_comparison.png` — comparison chart for accuracy and malignant precision, recall, and F1.
- `graphs/confusion_matrices.png` — combined view of all models' confusion matrices.
- `outputs/class_distribution.png`, `outputs/feature_distributions.png`, and `outputs/correlation_heatmap.png` — Part 1 visual exploration.
- `project_report.docx` — final project report with actual metrics, figures, limitations, and official documentation references.

### Actual holdout results

Values below are read from `outputs/comparison_results.csv`; rounded to four decimal places for display.

| Model | Accuracy | Malignant precision | Malignant recall | Malignant F1 | Benign precision | Benign recall | Benign F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9474 | 0.9286 | 0.9286 | 0.9286 | 0.9583 | 0.9583 | 0.9583 |
| Support Vector Machine (SVM) | **0.9825** | **0.9762** | **0.9762** | **0.9762** | **0.9861** | **0.9861** | **0.9861** |
| Gaussian Naive Bayes | 0.9386 | 0.9268 | 0.9048 | 0.9157 | 0.9452 | 0.9583 | 0.9517 |
| Gradient Boosting | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9467 | 0.9861 | 0.9660 |

SVM has the highest malignant-class F1-score and recall on this split. Its confusion matrix shows 41 of 42 malignant samples and 71 of 72 benign samples classified correctly. This is a result on one holdout set, not evidence of clinical validity or guaranteed performance on new populations.

## Limitations and future enhancements

The comparison uses one stratified 80/20 split and default or modest hand-selected model settings; it does not include cross-validation, uncertainty intervals, hyperparameter search, calibration, decision-threshold analysis, external validation, or a clinical workflow. The dataset is relatively small and class counts are not equal. Future educational work could add repeated stratified cross-validation, model tuning confined to training folds, confidence intervals, and additional classification metrics. Any medical application would require qualified clinical oversight and extensive independent validation.

## Official Scikit-learn references

- [Breast cancer dataset loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Toy datasets — Breast cancer Wisconsin diagnostic dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- [Train/test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [Feature standardization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
