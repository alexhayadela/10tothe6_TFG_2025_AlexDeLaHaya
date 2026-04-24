import numpy as np
import pandas as pd
import torch
import pytest
from models.neural.lstm import add_cyclic_dow, build_sequences, SequenceDataset


@pytest.fixture
def feature_df():
    """Small single-ticker feature DataFrame with a dow column."""
    n = 30
    np.random.seed(1)
    df = pd.DataFrame(
        np.random.randn(n, 4),
        columns=["feat_a", "feat_b", "feat_c", "dow"],
    )
    df["dow"] = np.tile(np.arange(5), n // 5 + 1)[:n].astype(float)
    return df


@pytest.fixture
def multi_ticker_seqs():
    """Three tickers, 20 rows each, 5 features + dow."""
    frames = []
    dates = pd.date_range("2023-01-02", periods=20, freq="B")
    for t in ["A.MC", "B.MC", "C.MC"]:
        np.random.seed(hash(t) % 2**31)
        df = pd.DataFrame(np.random.randn(20, 5), columns=[f"f{i}" for i in range(4)] + ["dow"])
        df["dow"] = np.tile(np.arange(5), 4)[:20].astype(float)
        df["ticker"] = t
        df["date"] = dates
        df["y"] = np.random.randint(0, 2, 20).astype(float)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_add_cyclic_dow_removes_dow(feature_df):
    result = add_cyclic_dow(feature_df)
    assert "dow" not in result.columns


def test_add_cyclic_dow_adds_sin_cos(feature_df):
    result = add_cyclic_dow(feature_df)
    assert "dow_sin" in result.columns
    assert "dow_cos" in result.columns


def test_add_cyclic_dow_sin_cos_range(feature_df):
    result = add_cyclic_dow(feature_df)
    assert (result["dow_sin"].between(-1, 1)).all()
    assert (result["dow_cos"].between(-1, 1)).all()


def test_add_cyclic_dow_column_count(feature_df):
    result = add_cyclic_dow(feature_df)
    # removes 1 (dow), adds 2 (dow_sin, dow_cos) -> net +1
    assert len(result.columns) == len(feature_df.columns) + 1


def test_build_sequences_shape(multi_ticker_seqs):
    seq_len = 5
    feature_cols = [f"f{i}" for i in range(4)] + ["dow"]
    X = multi_ticker_seqs[feature_cols]
    y = multi_ticker_seqs["y"]
    tickers = multi_ticker_seqs["ticker"]
    dates   = multi_ticker_seqs["date"]
    seqs, labs, last_dates = build_sequences(X, y, tickers, dates, seq_len)
    n_per_ticker = 20 - seq_len + 1
    assert seqs.shape == (3 * n_per_ticker, seq_len, len(feature_cols))


def test_build_sequences_last_dates_length(multi_ticker_seqs):
    seq_len = 5
    feature_cols = [f"f{i}" for i in range(4)] + ["dow"]
    X = multi_ticker_seqs[feature_cols]
    y = multi_ticker_seqs["y"]
    tickers = multi_ticker_seqs["ticker"]
    dates   = multi_ticker_seqs["date"]
    seqs, labs, last_dates = build_sequences(X, y, tickers, dates, seq_len)
    assert len(last_dates) == len(seqs)
    assert len(labs) == len(seqs)


def test_sequence_dataset_len():
    X = np.random.randn(10, 5, 3).astype(np.float32)
    y = np.random.randint(0, 2, 10).astype(np.float32)
    ds = SequenceDataset(X, y)
    assert len(ds) == 10


def test_sequence_dataset_getitem():
    X = np.random.randn(10, 5, 3).astype(np.float32)
    y = np.ones(10, dtype=np.float32)
    ds = SequenceDataset(X, y)
    x_item, y_item = ds[0]
    assert isinstance(x_item, torch.Tensor)
    assert isinstance(y_item, torch.Tensor)
    assert x_item.shape == (5, 3)
