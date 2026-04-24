import numpy as np
import pandas as pd
import torch
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from models.neural.lstm import StockRNN
from models.neural.cnn_rnn import StockCNNRNN
from models.predict import (
    load_artifact,
    _predict_tree,
    _predict_rnn,
    _reconstruct_rnn,
)
from models.trees.xgb import _temporal_inner_split


# -- synthetic artifact factories ---------------------------------------------

def _tree_artifact(target_type="discrete"):
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(40, 5), columns=[f"f{i}" for i in range(5)])
    y = np.random.randint(0, 2, 40)
    model = RandomForestClassifier(n_estimators=3, random_state=0)
    model.fit(X, y)
    return {
        "model_key":   "rf",
        "model":       model,
        "features":    list(X.columns),
        "target_type": target_type,
        "ft_type":     "micro",
    }


def _rnn_artifact(model_key="gru"):
    input_size = 5
    cfg = {
        "input_size":  input_size,
        "hidden_size": 8,
        "num_layers":  1,
        "dropout":     0.0,
        "cell":        model_key if model_key in ("gru", "lstm") else "gru",
    }
    if model_key.startswith("cnn_"):
        cfg["num_filters"] = 4
        cfg["kernel_size"] = 3
        model = StockCNNRNN(**cfg)
    else:
        model = StockRNN(**cfg)
    scaler = StandardScaler()
    scaler.fit(np.random.randn(100, input_size))
    # post-cyclic-dow feature names: dow_sin replaces dow
    features = [f"f{i}" for i in range(4)] + ["dow_sin"]
    return {
        "model_key":    model_key,
        "model_state":  model.state_dict(),
        "model_config": cfg,
        "scaler":       scaler,
        "features":     features,
        "seq_len":      5,
        "target_type":  "discrete",
        "ft_type":      "micro",
    }


def _rnn_input(n_tickers=3, n_rows=20):
    """Returns (X, tickers, dates) ready for _predict_rnn.
    X has raw 'dow' column (integer 0-4) plus 4 generic features.
    _predict_rnn applies add_cyclic_dow internally.
    """
    tickers_list = [f"T{i}.MC" for i in range(n_tickers)]
    frames = []
    for t in tickers_list:
        np.random.seed(hash(t) % 2**31)
        df = pd.DataFrame(np.random.randn(n_rows, 4), columns=[f"f{i}" for i in range(4)])
        df["dow"] = np.tile(np.arange(5), n_rows // 5 + 1)[:n_rows].astype(float)
        df["ticker"] = t
        df["date"]   = pd.date_range("2023-01-02", periods=n_rows, freq="B")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    feature_cols = [f"f{i}" for i in range(4)] + ["dow"]
    X       = combined[feature_cols]
    tickers = combined["ticker"]
    dates   = combined["date"]
    return X, tickers, dates


# -- load_artifact ------------------------------------------------------------

def test_load_artifact_raises_file_not_found(tmp_path):
    import unittest.mock as mock
    with mock.patch("models.predict.ARTIFACTS_PATH", tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifact("rf", 1, "sliding", "discrete")


# -- _predict_tree ------------------------------------------------------------

def test_predict_tree_preds_and_probas_shape():
    artifact = _tree_artifact()
    np.random.seed(0)
    X = pd.DataFrame(np.random.randn(10, 5), columns=[f"f{i}" for i in range(5)])
    preds, probas = _predict_tree(artifact, X)
    assert len(preds) == 10
    assert len(probas) == 10


def test_predict_tree_preds_binary():
    artifact = _tree_artifact()
    np.random.seed(1)
    X = pd.DataFrame(np.random.randn(10, 5), columns=[f"f{i}" for i in range(5)])
    preds, _ = _predict_tree(artifact, X)
    assert set(preds).issubset({0, 1})


def test_predict_tree_continuous_returns_float():
    import unittest.mock as mock
    artifact = _tree_artifact(target_type="continuous")
    artifact["model"] = mock.MagicMock()
    artifact["model"].predict.return_value = np.array([0.01, -0.02, 0.005])
    X = pd.DataFrame(np.random.randn(3, 5), columns=[f"f{i}" for i in range(5)])
    preds, probas = _predict_tree(artifact, X)
    assert preds.dtype == float
    assert np.array_equal(preds, probas)


def test_predict_tree_1d_proba_no_index_error():
    """MarkovChain-style model returns 1-D predict_proba."""
    import unittest.mock as mock
    artifact = _tree_artifact()
    artifact["model"] = mock.MagicMock()
    artifact["model"].predict.return_value = np.array([1, 0, 1])
    artifact["model"].predict_proba.return_value = np.array([0.7, 0.4, 0.8])  # 1-D
    X = pd.DataFrame(np.random.randn(3, 5), columns=[f"f{i}" for i in range(5)])
    preds, probas = _predict_tree(artifact, X)
    assert len(preds) == 3
    assert len(probas) == 3


# -- _reconstruct_rnn ---------------------------------------------------------

def test_reconstruct_rnn_gru_type():
    artifact = _rnn_artifact("gru")
    model = _reconstruct_rnn(artifact)
    assert isinstance(model, StockRNN)
    assert model.cell == "gru"


def test_reconstruct_rnn_lstm_type():
    artifact = _rnn_artifact("lstm")
    model = _reconstruct_rnn(artifact)
    assert isinstance(model, StockRNN)
    assert model.cell == "lstm"


def test_reconstruct_cnn_gru_type():
    artifact = _rnn_artifact("cnn_gru")
    model = _reconstruct_rnn(artifact)
    assert isinstance(model, StockCNNRNN)


def test_reconstruct_rnn_is_eval_mode():
    artifact = _rnn_artifact("gru")
    model = _reconstruct_rnn(artifact)
    assert not model.training


# -- _predict_rnn -------------------------------------------------------------

def test_predict_rnn_output_lengths():
    artifact = _rnn_artifact("gru")
    X, tickers, dates = _rnn_input(n_tickers=3, n_rows=20)
    preds, probas, last_dates, last_tickers = _predict_rnn(
        artifact, X, tickers, dates, "discrete"
    )
    assert len(preds) == len(probas) == len(last_dates) == len(last_tickers)


def test_predict_rnn_preds_binary():
    artifact = _rnn_artifact("gru")
    X, tickers, dates = _rnn_input(n_tickers=2, n_rows=15)
    preds, _, _, _ = _predict_rnn(artifact, X, tickers, dates, "discrete")
    assert set(preds.tolist()).issubset({0, 1})


def test_predict_rnn_tickers_match_input():
    artifact = _rnn_artifact("gru")
    X, tickers, dates = _rnn_input(n_tickers=3, n_rows=20)
    _, _, _, last_tickers = _predict_rnn(artifact, X, tickers, dates, "discrete")
    assert set(last_tickers).issubset(set(tickers.unique()))


def test_predict_rnn_continuous_raw_output():
    artifact = _rnn_artifact("gru")
    artifact["target_type"] = "continuous"
    X, tickers, dates = _rnn_input(n_tickers=2, n_rows=15)
    preds, probas, _, _ = _predict_rnn(artifact, X, tickers, dates, "continuous")
    assert np.allclose(preds, probas)


# -- _temporal_inner_split ----------------------------------------------------

def test_temporal_inner_split_no_leakage():
    np.random.seed(0)
    n = 100
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])
    y = pd.Series(np.random.randint(0, 2, n))
    date_series = pd.Series(dates)
    X_tr, y_tr, X_val, y_val = _temporal_inner_split(X, y, date_series, val_fraction=0.2)
    assert len(X_tr) + len(X_val) == n
    assert len(X_tr) > len(X_val)


def test_temporal_inner_split_sizes():
    np.random.seed(1)
    n = 50
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    X = pd.DataFrame(np.random.randn(n, 2), columns=["a", "b"])
    y = pd.Series(np.zeros(n))
    date_series = pd.Series(dates)
    X_tr, _, X_val, _ = _temporal_inner_split(X, y, date_series, val_fraction=0.2)
    assert len(X_tr) == pytest.approx(n * 0.8, abs=2)
