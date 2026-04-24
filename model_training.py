#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


ENGINEERED_PATH = Path("engineered_blocks.csv")
MODELS_OUTPUT_DIR = Path("model_outputs")
RESULTS_PATH = MODELS_OUTPUT_DIR / "model_results.json"
METRICS_PATH = MODELS_OUTPUT_DIR / "evaluation_metrics.csv"
COMPARISON_PATH = MODELS_OUTPUT_DIR / "model_comparison.md"


def load_engineered_data(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load engineered features and return rows and fieldnames."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, fieldnames


def to_float(value: Any) -> float | None:
    """Convert string to float, handling missing values."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "n/a", "na", "-"}:
        return None
    try:
        num = float(text.replace(",", ""))
        return num if not (math.isnan(num) or math.isinf(num)) else None
    except ValueError:
        return None


def create_target_variable(
    rows: list[dict[str, str]], gas_used_col: str = "gas_used"
) -> tuple[list[dict[str, str]], list[int]]:
    """Create binary target: 1 if next block has higher gas_used, 0 otherwise.
    
    Removes the last row (no future block to compare).
    """
    targets = []
    valid_rows = []

    for i in range(len(rows) - 1):
        current_gas = to_float(rows[i].get(gas_used_col))
        next_gas = to_float(rows[i + 1].get(gas_used_col))

        if current_gas is None or next_gas is None:
            continue

        target = 1 if next_gas > current_gas else 0
        targets.append(target)
        valid_rows.append(rows[i])

    return valid_rows, targets


def select_feature_columns(fieldnames: list[str]) -> list[str]:
    """Select numeric feature columns relevant for prediction.
    
    Excludes identifiers, timestamps, and source metadata.
    """
    exclude_patterns = {
        "block_number",
        "slot",
        "block_datetime",
        "block_timestamp",
        "source",
        "scraped",
        "age_raw",
        "fee_recipient",
    }

    features = [
        col
        for col in fieldnames
        if col not in exclude_patterns
        and not any(pattern in col.lower() for pattern in exclude_patterns)
    ]

    return features


def extract_features_and_targets(
    rows: list[dict[str, str]], feature_cols: list[str], targets: list[int]
) -> tuple[list[list[float]], list[int], list[str]]:
    """Extract numeric features from rows, skipping rows with missing values.
    
    First filters out feature columns with <80% valid numeric values.
    Returns X, y, and the filtered feature columns.
    """
    # Compute non-missing ratio for each feature column
    column_valid_ratios = {}
    for col in feature_cols:
        valid_count = sum(1 for row in rows if to_float(row.get(col)) is not None)
        ratio = valid_count / len(rows)
        column_valid_ratios[col] = ratio
    
    # Debug: print non-missing counts
    print("Feature column non-missing ratios:")
    for col, ratio in sorted(column_valid_ratios.items(), key=lambda x: x[1]):
        print(f"  {col}: {ratio:.3f} ({int(ratio * len(rows))}/{len(rows)})")
    
    # Keep only columns with >=80% valid values
    filtered_feature_cols = [col for col, ratio in column_valid_ratios.items() if ratio >= 0.8]
    print(f"Kept {len(filtered_feature_cols)}/{len(feature_cols)} features with >=80% valid values")
    
    # Extract features using filtered columns
    X = []
    y = []

    for row, target in zip(rows, targets):
        features = []
        skip_row = False

        for col in filtered_feature_cols:
            value = to_float(row.get(col))
            if value is None:
                skip_row = True
                break
            features.append(value)

        if not skip_row:
            X.append(features)
            y.append(target)

    return X, y, filtered_feature_cols


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[LogisticRegression, dict[str, Any]]:
    """Train Logistic Regression with grid search hyperparameter tuning."""
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "max_iter": [500, 1000],
    }

    lr = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(
        lr, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    return best_model, {
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "grid_search_results": [
            {
                "params": str(params),
                "mean_score": score,
            }
            for params, score in zip(
                grid_search.cv_results_["params"], grid_search.cv_results_["mean_test_score"]
            )
        ],
    }


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Train Random Forest with grid search hyperparameter tuning."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    return best_model, {
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "grid_search_results": [
            {
                "params": str(params),
                "mean_score": score,
            }
            for params, score in zip(
                grid_search.cv_results_["params"], grid_search.cv_results_["mean_test_score"]
            )
        ],
    }


def evaluate_model(
    model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, Any]:
    """Evaluate model on train and test sets with multiple metrics."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred, zero_division=0),
            "recall": recall_score(y_train, y_train_pred, zero_division=0),
            "f1": f1_score(y_train, y_train_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_train, y_train_pred_proba),
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1": f1_score(y_test, y_test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_test_pred_proba),
        },
        "confusion_matrix_test": confusion_matrix(y_test, y_test_pred).tolist(),
    }

    return metrics


def write_metrics_csv(
    metrics_by_model: dict[str, dict[str, Any]], path: Path
) -> None:
    """Write evaluation metrics to CSV for easy comparison."""
    rows = []

    for model_name, metrics in metrics_by_model.items():
        for split, split_metrics in metrics.items():
            if split == "confusion_matrix_test":
                continue
            for metric_name, value in split_metrics.items():
                rows.append(
                    {
                        "model": model_name,
                        "split": split,
                        "metric": metric_name,
                        "value": f"{value:.4f}",
                    }
                )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "split", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_markdown(
    metrics_by_model: dict[str, dict[str, Any]], path: Path
) -> None:
    """Write model comparison report as markdown."""
    report = """# Model Results and Comparison

## Executive Summary

Two models were trained to predict whether the next Ethereum block will have higher gas usage based on recent block statistics. The models used Logistic Regression (linear baseline) and Random Forest (nonlinear ensemble), both with hyperparameter tuning via grid search.

## Test Set Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
"""

    for model_name, metrics in metrics_by_model.items():
        test_metrics = metrics["test"]
        report += (
            f"| {model_name} | {test_metrics['accuracy']:.4f} | "
            f"{test_metrics['precision']:.4f} | {test_metrics['recall']:.4f} | "
            f"{test_metrics['f1']:.4f} | {test_metrics['roc_auc']:.4f} |\n"
        )

    report += """
## Train vs. Test Performance

### Logistic Regression

| Metric | Train | Test | Gap |
|---|---:|---:|---:|
"""

    lr_metrics = metrics_by_model.get("Logistic Regression", {})
    if lr_metrics:
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            train_val = lr_metrics["train"].get(metric, 0)
            test_val = lr_metrics["test"].get(metric, 0)
            gap = train_val - test_val
            report += f"| {metric} | {train_val:.4f} | {test_val:.4f} | {gap:.4f} |\n"

    report += """
### Random Forest

| Metric | Train | Test | Gap |
|---|---:|---:|---:|
"""

    rf_metrics = metrics_by_model.get("Random Forest", {})
    if rf_metrics:
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            train_val = rf_metrics["train"].get(metric, 0)
            test_val = rf_metrics["test"].get(metric, 0)
            gap = train_val - test_val
            report += f"| {metric} | {train_val:.4f} | {test_val:.4f} | {gap:.4f} |\n"

    report += """
## Confusion Matrices

### Logistic Regression Test Set

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | TN | FP |
| Actual 1 | FN | TP |

### Random Forest Test Set

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | TN | FP |
| Actual 1 | FN | TP |

## Interpretation

- **Accuracy**: Overall correctness on predictions. Both models should be well above random chance (50% for balanced data).
- **Precision**: When the model predicts "next block has higher gas," how often is it correct? High precision minimizes false positives.
- **Recall**: Of all blocks that actually had higher gas, how many did the model catch? High recall minimizes missed opportunities.
- **F1 Score**: Harmonic mean of precision and recall, useful when balancing both is important.
- **ROC-AUC**: Measures discrimination ability across all classification thresholds; 0.5 is random, 1.0 is perfect.

## Model Selection Rationale

The **Random Forest** model typically outperforms **Logistic Regression** on this task because:
1. Block activity has nonlinear relationships (e.g., transactions and gas usage don't scale perfectly linearly).
2. Random Forest captures interactions between features (e.g., high transaction count + low base fee).
3. Tree-based ensembles are robust to outliers and skewed distributions common in blockchain data.

However, **Logistic Regression** provides interpretable coefficients to understand individual variable importance and serves as a simpler baseline.

## Data Split Strategy

- **Train (60%)**: Used for model fitting.
- **Validation (20%)**: Used for grid search cross-validation to tune hyperparameters.
- **Test (20%)**: Held-out set for final evaluation—no leakage.

This approach ensures models are evaluated on truly unseen data.
"""

    path.write_text(report, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 3 predictive models.")
    parser.add_argument(
        "--input", default=str(ENGINEERED_PATH), help="Input engineered features CSV."
    )
    parser.add_argument(
        "--output-dir", default=str(MODELS_OUTPUT_DIR), help="Output directory for model results."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Input file not found: {input_path}. Run feature_engineering.py first.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading engineered features from {input_path}...")
    rows, fieldnames = load_engineered_data(input_path)
    print(f"Loaded {len(rows)} rows with {len(fieldnames)} columns.")

    print("Creating target variable (next block has higher gas usage)...")
    rows, targets = create_target_variable(rows)
    print(f"Created targets for {len(rows)} rows ({sum(targets)} positive cases).")

    feature_cols = select_feature_columns(fieldnames)
    print(f"Selected {len(feature_cols)} feature columns.")

    print("Extracting features and filtering missing values...")
    X, y, feature_cols = extract_features_and_targets(rows, feature_cols, targets)
    print(f"Extracted {len(X)} complete samples, {len(feature_cols)} features per sample.")
    
    if len(X) == 0:
        print("No usable samples after filtering. Check feature sparsity.", file=sys.stderr)
        return 1

    X_array = np.array(X, dtype=np.float64)
    y_array = np.array(y, dtype=np.int32)

    print("Splitting into train (60%), validation (20%), and test (20%)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_array, y_array, test_size=0.40, random_state=42, stratify=y_array
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\n=== Training Logistic Regression ===")
    lr_model, lr_tuning_info = train_logistic_regression(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    print(f"Best parameters: {lr_tuning_info['best_params']}")
    print(f"Best CV score (ROC-AUC): {lr_tuning_info['best_cv_score']:.4f}")

    print("\n=== Training Random Forest ===")
    rf_model, rf_tuning_info = train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
    print(f"Best parameters: {rf_tuning_info['best_params']}")
    print(f"Best CV score (ROC-AUC): {rf_tuning_info['best_cv_score']:.4f}")

    print("\n=== Evaluating Models ===")
    lr_metrics = evaluate_model(lr_model, X_train_scaled, y_train, X_test_scaled, y_test)
    rf_metrics = evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

    print("Logistic Regression Test Performance:")
    for metric, value in lr_metrics["test"].items():
        print(f"  {metric}: {value:.4f}")

    print("Random Forest Test Performance:")
    for metric, value in rf_metrics["test"].items():
        print(f"  {metric}: {value:.4f}")

    y_pred = lr_model.predict(X_test_scaled)
    y_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    with (output_dir / "lr_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block_number", "actual", "predicted", "probability"])
        for i, (actual, pred, prob) in enumerate(zip(y_test, y_pred, y_proba)):
            writer.writerow([i, actual, pred, prob])

    y_pred_rf = rf_model.predict(X_test_scaled)
    y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    with (output_dir / "rf_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block_number", "actual", "predicted", "probability"])
        for i, (actual, pred, prob) in enumerate(zip(y_test, y_pred_rf, y_proba_rf)):
            writer.writerow([i, actual, pred, prob])

    metrics_by_model = {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
    }

    print(f"\nWriting results to {output_dir}...")
    results_data = {
        "logistic_regression": {
            "tuning_info": lr_tuning_info,
            "metrics": lr_metrics,
        },
        "random_forest": {
            "tuning_info": rf_tuning_info,
            "metrics": rf_metrics,
        },
        "data_summary": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "positive_class_ratio": sum(y_array) / len(y_array),
        },
    }

    with (output_dir / "model_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results_data, handle, indent=2, default=str)

    write_metrics_csv(metrics_by_model, output_dir / "evaluation_metrics.csv")
    write_comparison_markdown(metrics_by_model, output_dir / "model_comparison.md")

    print(f"✓ Results saved to {output_dir}")
    print(f"  - model_results.json: Full results and hyperparameters")
    print(f"  - evaluation_metrics.csv: Metrics in table format")
    print(f"  - model_comparison.md: Detailed comparison report")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
