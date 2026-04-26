"""
Shared evaluation metrics for all models (RF, XGBoost, GRU, LSTM).

Every model must call evaluate_model() with identical arguments to guarantee
fair comparison. No model-specific adjustments to metrics are allowed.
Primary metric for model selection: balanced_accuracy (see decisions/rf_decisions.md).
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(y_true, y_pred, y_proba, model_name: str = "") -> dict:
    """Compute all evaluation metrics for binary direction classification.

    Parameters
    ----------
    y_true     : array-like of int (0=down, 1=up)
    y_pred     : array-like of int, hard predictions
    y_proba    : array-like of float, predicted P(class=1)
    model_name : str label, stored in the returned dict

    Returns
    -------
    dict with keys: model, accuracy, balanced_accuracy (PRIMARY), roc_auc,
    log_loss, mcc, mean_predicted_prob, pred_positive_rate
    """
    y_proba_safe = np.clip(y_proba, 1e-7, 1 - 1e-7)

    return {
        "model":               model_name,
        "accuracy":            float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy":   float(balanced_accuracy_score(y_true, y_pred)),  # PRIMARY
        "roc_auc":             float(roc_auc_score(y_true, y_proba_safe)),
        "log_loss":            float(log_loss(y_true, y_proba_safe)),
        "mcc":                 float(matthews_corrcoef(y_true, y_pred)),
        "mean_predicted_prob": float(np.mean(y_proba)),
        "pred_positive_rate":  float(np.mean(y_pred)),
    }


def evaluate_regression(y_true, y_pred, model_name: str = "") -> dict:
    """Compute regression evaluation metrics for continuous return prediction.

    Parameters
    ----------
    y_true     : array-like of float, actual future log returns
    y_pred     : array-like of float, predicted future log returns
    model_name : str label, stored in the returned dict

    Returns
    -------
    dict with keys: model, mae, rmse, r2, directional_accuracy, ic (PRIMARY)
    See decisions/continuous_target_decisions.md for metric rationale.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ic, _ = spearmanr(y_true, y_pred)

    return {
        "model":               model_name,
        "mae":                 float(mean_absolute_error(y_true, y_pred)),
        "rmse":                float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":                  float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "ic":                  float(ic) if np.isfinite(ic) else 0.0,  # PRIMARY
        "mean_abs_pred":       float(np.mean(np.abs(y_pred))),
    }


def print_regression_metrics(metrics: dict):
    """Print a regression metrics dict in a readable format."""
    print(f"\n{'-' * 45}")
    if metrics.get("model"):
        print(f"  Model             : {metrics['model']}")
    print(f"  MAE               : {metrics['mae']:.6f}")
    print(f"  RMSE              : {metrics['rmse']:.6f}")
    print(f"  R²                : {metrics['r2']:.6f}")
    print(f"  Directional acc   : {metrics['directional_accuracy']:.4f}")
    print(f"  IC (Spearman)     : {metrics['ic']:.4f}  <- primary")
    print(f"{'-' * 45}\n")


def print_metrics(metrics: dict, report: bool = False, y_true=None, y_pred=None):
    """Print a metrics dict in a readable format.

    Pass report=True plus y_true / y_pred to also print the full
    per-class classification report (precision, recall, F1).
    """
    print(f"\n{'-' * 45}")
    if metrics.get("model"):
        print(f"  Model             : {metrics['model']}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}  <- primary")
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"  Log Loss          : {metrics['log_loss']:.4f}")
    print(f"  MCC               : {metrics['mcc']:.4f}")
    print(f"  Mean pred prob    : {metrics['mean_predicted_prob']:.4f}")
    print(f"  Pred positive rate: {metrics['pred_positive_rate']:.4f}")
    print(f"{'-' * 45}\n")

    if report and y_true is not None and y_pred is not None:
        print(classification_report(y_true, y_pred, digits=4))
