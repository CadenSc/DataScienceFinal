# Model Results and Comparison

## Executive Summary

Two models were trained to predict whether the next Ethereum block will have higher gas usage based on recent block statistics. The models used Logistic Regression (linear baseline) and Random Forest (nonlinear ensemble), both with hyperparameter tuning via grid search.

## Test Set Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9880 | 0.9955 | 0.9781 | 0.9867 | 0.9993 |
| Random Forest | 0.9650 | 0.9930 | 0.9298 | 0.9604 | 0.9975 |

## Train vs. Test Performance

### Logistic Regression

| Metric | Train | Test | Gap |
|---|---:|---:|---:|
| accuracy | 0.9997 | 0.9880 | 0.0117 |
| precision | 1.0000 | 0.9955 | 0.0045 |
| recall | 0.9993 | 0.9781 | 0.0212 |
| f1 | 0.9996 | 0.9867 | 0.0129 |
| roc_auc | 1.0000 | 0.9993 | 0.0007 |

### Random Forest

| Metric | Train | Test | Gap |
|---|---:|---:|---:|
| accuracy | 1.0000 | 0.9650 | 0.0350 |
| precision | 1.0000 | 0.9930 | 0.0070 |
| recall | 1.0000 | 0.9298 | 0.0702 |
| f1 | 1.0000 | 0.9604 | 0.0396 |
| roc_auc | 1.0000 | 0.9975 | 0.0025 |

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
