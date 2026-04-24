import numpy as np
import pytest
from models.evaluate import evaluate_model, evaluate_regression


def test_evaluate_model_keys():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    probas = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.2, 0.8])
    result = evaluate_model(y, preds, probas)
    expected_keys = {"model", "accuracy", "balanced_accuracy", "roc_auc",
                     "log_loss", "mcc", "mean_predicted_prob", "pred_positive_rate"}
    assert set(result.keys()) == expected_keys


def test_evaluate_model_perfect_predictions():
    y = np.array([0, 1, 0, 1, 0, 1])
    preds = y.copy()
    probas = np.where(y == 1, 0.99, 0.01).astype(float)
    result = evaluate_model(y, preds, probas)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["mcc"] == pytest.approx(1.0)


def test_evaluate_model_values_in_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    preds = rng.integers(0, 2, 100)
    probas = rng.uniform(0, 1, 100)
    result = evaluate_model(y, preds, probas)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["balanced_accuracy"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert result["log_loss"] >= 0.0
    assert -1.0 <= result["mcc"] <= 1.0


def test_evaluate_regression_keys():
    y = np.array([0.01, -0.02, 0.005, 0.03, -0.01])
    preds = np.array([0.008, -0.015, 0.006, 0.025, -0.008])
    result = evaluate_regression(y, preds)
    expected_keys = {"model", "mae", "rmse", "r2", "directional_accuracy", "ic"}
    assert set(result.keys()) == expected_keys


def test_evaluate_regression_perfect():
    y = np.array([0.01, -0.02, 0.005, 0.03, -0.01])
    result = evaluate_regression(y, y.copy())
    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)
    assert result["directional_accuracy"] == pytest.approx(1.0)


def test_evaluate_regression_directional_accuracy():
    y    = np.array([0.01, 0.02, -0.01, -0.02])
    pred = np.array([0.005, 0.015, -0.005, -0.015])
    result = evaluate_regression(y, pred)
    assert result["directional_accuracy"] == pytest.approx(1.0)


def test_evaluate_regression_wrong_direction():
    y    = np.array([0.01, 0.02, -0.01, -0.02])
    pred = -y
    result = evaluate_regression(y, pred)
    assert result["directional_accuracy"] == pytest.approx(0.0)
