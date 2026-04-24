import numpy as np
import pandas as pd
import pytest
from models.trees.features import (
    assert_columns,
    rolling_slope,
    rsi,
    micro_features,
    horizon_features,
    target_feature,
    cross_micro_features,
    ml_ready,
)


@pytest.fixture
def single_ohlcv():
    n = 80
    np.random.seed(7)
    close = 10.0 + np.cumsum(np.random.randn(n) * 0.1)
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "ticker": "X.MC",
        "date":   pd.date_range("2022-01-03", periods=n, freq="B"),
        "open":   close,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": np.random.randint(100_000, 500_000, n).astype(float),
    })


@pytest.fixture
def multi_ohlcv():
    frames = []
    for t in ["A.MC", "B.MC", "C.MC"]:
        np.random.seed(hash(t) % 2**31)
        n = 80
        close = 10.0 + np.cumsum(np.random.randn(n) * 0.1)
        close = np.maximum(close, 1.0)
        frames.append(pd.DataFrame({
            "ticker": t,
            "date":   pd.date_range("2022-01-03", periods=n, freq="B"),
            "open": close, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": np.random.randint(100_000, 500_000, n).astype(float),
        }))
    return pd.concat(frames, ignore_index=True)


def test_assert_columns_passes():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert_columns(df, ["a", "b"])  # no raise


def test_assert_columns_raises_on_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(AssertionError):
        assert_columns(df, ["a", "missing_col"])


def test_rolling_slope_length(single_ohlcv):
    result = rolling_slope(single_ohlcv["close"], window=5)
    assert len(result) == len(single_ohlcv)


def test_rolling_slope_constant_is_zero():
    s = pd.Series([5.0] * 20)
    slope = rolling_slope(s, window=5)
    valid = slope.dropna()
    assert (valid.abs() < 1e-10).all()


def test_rsi_range(single_ohlcv):
    result = rsi(single_ohlcv["close"], window=14)
    valid = result.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_all_gains():
    s = pd.Series(range(1, 30, 1), dtype=float)
    result = rsi(s, window=14).dropna()
    assert (result > 90).all()


def test_micro_features_expected_columns(single_ohlcv):
    result = micro_features(single_ohlcv)
    for col in ["log_ret_1", "vol_20", "rsi_14", "macd_hist", "atr_pct"]:
        assert col in result.columns, f"Missing column: {col}"


def test_micro_features_no_inf(single_ohlcv):
    result = micro_features(single_ohlcv)
    numeric = result.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values).any()


def test_horizon_features_h1_adds_dow(single_ohlcv):
    with_micro = micro_features(single_ohlcv)
    result = horizon_features(with_micro, horizon=1)
    assert "dow" in result.columns


def test_horizon_features_h5_adds_momentum(single_ohlcv):
    with_micro = micro_features(single_ohlcv)
    result = horizon_features(with_micro, horizon=5)
    assert "mom_12_1" in result.columns


def test_target_feature_direction_binary(single_ohlcv):
    result = target_feature(single_ohlcv, horizon=1)
    valid = result["target"].dropna()
    assert set(valid.unique()).issubset({0, 1})


def test_target_feature_future_log_ret_is_float(single_ohlcv):
    result = target_feature(single_ohlcv, horizon=1)
    assert result["future_log_ret"].dtype == float


def test_cross_micro_features_adds_breadth(multi_ohlcv):
    frames = []
    for t, g in multi_ohlcv.groupby("ticker"):
        g = g.copy().sort_values("date")
        g["log_ret_1"] = np.log(g["close"] / g["close"].shift(1))
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    result = cross_micro_features(df)
    assert "ibx_breadth_10d" in result.columns


def test_ml_ready_returns_five_tuple(multi_ohlcv):
    result = ml_ready(horizon=1, df_micro=multi_ohlcv, df_macro=None, ft_type="micro")
    assert len(result) == 5


def test_ml_ready_X_no_nan(multi_ohlcv):
    _, X, _, mask, _ = ml_ready(horizon=1, df_micro=multi_ohlcv, df_macro=None, ft_type="micro")
    assert not X.isnull().any().any()


def test_ml_ready_y_binary(multi_ohlcv):
    _, _, y, mask, _ = ml_ready(horizon=1, df_micro=multi_ohlcv, df_macro=None, ft_type="micro")
    assert set(y.unique()).issubset({0, 1})


def test_ml_ready_mask_aligns_X_and_df(multi_ohlcv):
    df, X, y, mask, _ = ml_ready(horizon=1, df_micro=multi_ohlcv, df_macro=None, ft_type="micro")
    assert len(X) == mask.sum()
    assert len(y) == mask.sum()
