import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dates_array():
    """1000 sequential trading dates as numpy array."""
    base = pd.date_range("2020-01-02", periods=1000, freq="B")
    return base.to_numpy()


@pytest.fixture
def single_ticker_ohlcv():
    """60-row single-ticker OHLCV DataFrame."""
    n = 60
    np.random.seed(42)
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.1)
    close = np.maximum(close, 1.0)
    df = pd.DataFrame({
        "ticker": "TEST.MC",
        "date":   pd.date_range("2023-01-02", periods=n, freq="B"),
        "open":   close * (1 + np.random.randn(n) * 0.005),
        "high":   close * (1 + np.abs(np.random.randn(n) * 0.01)),
        "low":    close * (1 - np.abs(np.random.randn(n) * 0.01)),
        "close":  close,
        "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    })
    return df


@pytest.fixture
def multi_ticker_ohlcv():
    """60-row × 3 tickers OHLCV DataFrame."""
    frames = []
    for ticker in ["AAA.MC", "BBB.MC", "CCC.MC"]:
        np.random.seed(hash(ticker) % 2**31)
        n = 60
        close = 10.0 + np.cumsum(np.random.randn(n) * 0.1)
        close = np.maximum(close, 1.0)
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "date":   pd.date_range("2023-01-02", periods=n, freq="B"),
            "open":   close * (1 + np.random.randn(n) * 0.005),
            "high":   close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "low":    close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "close":  close,
            "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def macro_ohlcv():
    """60-row macro DataFrame with IBEX, SP500, VIX tickers."""
    frames = []
    specs = [("^IBEX", 9000, 200), ("^GSPC", 4500, 50), ("^VIX", 20, 3)]
    for ticker, base_price, vol in specs:
        np.random.seed(hash(ticker) % 2**31)
        n = 60
        close = base_price + np.cumsum(np.random.randn(n) * vol * 0.01)
        close = np.maximum(close, 1.0)
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "date":   pd.date_range("2023-01-02", periods=n, freq="B"),
            "open":   close,
            "high":   close * 1.005,
            "low":    close * 0.995,
            "close":  close,
            "volume": 0.0,
        }))
    return pd.concat(frames, ignore_index=True)
