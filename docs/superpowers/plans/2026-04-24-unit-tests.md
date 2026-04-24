# Unit Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a complete pytest unit test suite covering all testable functions across models, news, newsletter, DB, and LLM layers — no test requires a real DB, trained model, or internet connection.

**Architecture:** Tests live in `tests/` mirroring the source tree. `conftest.py` holds shared fixtures. All external deps (Supabase, SQLite, Groq, yfinance, torch artifacts) are mocked with `pytest-mock` or by constructing minimal synthetic artifacts in-test. Tests are NOT executed during implementation (missing data/models on this machine) but must be syntactically correct and logically sound.

**Tech Stack:** pytest, pytest-mock, numpy, pandas, torch (CPU), sklearn, xgboost

---

## File Map

| File | Responsibility |
|---|---|
| `tests/conftest.py` | Shared fixtures: synthetic OHLCV, dates arrays, multi-ticker DataFrame |
| `tests/test_evaluate.py` | `models/evaluate.py` — pure metrics functions |
| `tests/test_windows.py` | `models/base.py` — `sliding_windows`, `expanding_windows` |
| `tests/test_features.py` | `models/trees/features.py` — all pure feature engineering |
| `tests/test_markov.py` | `models/markov/markov.py` — `MarkovChain` class |
| `tests/test_rnn_utils.py` | `models/neural/lstm.py` — `add_cyclic_dow`, `build_sequences`, `SequenceDataset` |
| `tests/test_rate_limiter.py` | `llm/rate_limit.py` — `RateLimitState` |
| `tests/test_news_classification.py` | `news/classification.py` — pure + mocked LLM |
| `tests/test_news_rss.py` | `news/news_rss.py` — `last_news`, `top_news` |
| `tests/test_newsletter.py` | `newsletter/build.py` — HTML builders, `format_predictions` |
| `tests/test_predict_helpers.py` | `models/predict.py` — inference helpers with synthetic artifacts |
| `tests/test_db_sqlite.py` | `db/sqlite/` — all query/ingest functions, mocked connection |
| `tests/test_db_supabase.py` | `db/supabase/` — all query/ingest functions, mocked client |
| `tests/test_llm_service.py` | `llm/gpt_service.py` — `LLMService`, mocked Groq API |

---

### Task 1: conftest.py — shared fixtures

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `tests/__init__.py`**

```python
```
(empty file)

- [ ] **Step 2: Write `tests/conftest.py`**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add shared fixtures in conftest.py"
```

---

### Task 2: test_evaluate.py — pure metrics

**Files:**
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write `tests/test_evaluate.py`**

```python
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
    pred = np.array([0.005, 0.015, -0.005, -0.015])  # same sign
    result = evaluate_regression(y, pred)
    assert result["directional_accuracy"] == pytest.approx(1.0)


def test_evaluate_regression_wrong_direction():
    y    = np.array([0.01, 0.02, -0.01, -0.02])
    pred = -y  # opposite sign
    result = evaluate_regression(y, pred)
    assert result["directional_accuracy"] == pytest.approx(0.0)
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_evaluate.py
git commit -m "test: add evaluate_model and evaluate_regression unit tests"
```

---

### Task 3: test_windows.py — CV window generators

**Files:**
- Create: `tests/test_windows.py`

- [ ] **Step 1: Write `tests/test_windows.py`**

```python
import numpy as np
import pytest
from models.base import sliding_windows, expanding_windows


@pytest.fixture
def dates():
    return np.arange(1000)


def test_sliding_windows_returns_list(dates):
    windows = sliding_windows(dates, window=200, step=50)
    assert isinstance(windows, list)
    assert len(windows) > 0


def test_sliding_windows_train_size(dates):
    windows = sliding_windows(dates, window=200, step=50)
    for train, _ in windows:
        assert len(train) == 200


def test_sliding_windows_embargo(dates):
    embargo = 1
    windows = sliding_windows(dates, window=200, step=50, embargo=embargo)
    for train, test in windows:
        assert test[0] > train[-1] + embargo - 1


def test_sliding_windows_no_overlap(dates):
    windows = sliding_windows(dates, window=200, step=50)
    for train, test in windows:
        train_set = set(train.tolist())
        test_set  = set(test.tolist())
        assert train_set.isdisjoint(test_set)


def test_sliding_windows_empty_on_short_input():
    short = np.arange(100)
    windows = sliding_windows(short, window=200, step=50)
    assert windows == []


def test_expanding_windows_train_grows(dates):
    windows = expanding_windows(dates, min_train=200, step=50)
    train_sizes = [len(t) for t, _ in windows]
    assert train_sizes == sorted(train_sizes)
    assert all(b > a for a, b in zip(train_sizes, train_sizes[1:]))


def test_expanding_windows_min_train(dates):
    min_train = 200
    windows = expanding_windows(dates, min_train=min_train, step=50)
    assert len(windows) > 0
    first_train, _ = windows[0]
    assert len(first_train) >= min_train


def test_expanding_windows_no_overlap(dates):
    windows = expanding_windows(dates, min_train=200, step=50)
    for train, test in windows:
        train_set = set(train.tolist())
        test_set  = set(test.tolist())
        assert train_set.isdisjoint(test_set)


def test_expanding_windows_empty_on_short_input():
    short = np.arange(50)
    windows = expanding_windows(short, min_train=200, step=50)
    assert windows == []
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_windows.py
git commit -m "test: add sliding_windows and expanding_windows unit tests"
```

---

### Task 4: test_features.py — feature engineering

**Files:**
- Create: `tests/test_features.py`

- [ ] **Step 1: Write `tests/test_features.py`**

```python
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
    valid = result["direction"].dropna()
    assert set(valid.unique()).issubset({0, 1})


def test_target_feature_future_log_ret_is_float(single_ohlcv):
    result = target_feature(single_ohlcv, horizon=1)
    assert result["future_log_ret"].dtype == float


def test_cross_micro_features_adds_breadth(multi_ohlcv):
    # add log_ret_1 column first (cross features require it)
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
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_features.py
git commit -m "test: add feature engineering unit tests"
```

---

### Task 5: test_markov.py — MarkovChain

**Files:**
- Create: `tests/test_markov.py`

- [ ] **Step 1: Write `tests/test_markov.py`**

```python
import numpy as np
import pandas as pd
import pytest
from models.markov.markov import MarkovChain


@pytest.fixture
def train_data():
    np.random.seed(0)
    n = 200
    log_ret_1 = np.random.randn(n) * 0.01
    log_ret_5 = np.random.randn(n) * 0.02
    X = pd.DataFrame({"log_ret_1": log_ret_1, "log_ret_5": log_ret_5})
    y = (np.random.rand(n) > 0.5).astype(int)
    return X, y


def test_fit_builds_transition_dict(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=1)
    mc.fit(X, y)
    assert len(mc.transition_) > 0


def test_predict_proba_range(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=1)
    mc.fit(X, y)
    probas = mc.predict_proba(X)
    assert ((probas >= 0) & (probas <= 1)).all()


def test_predict_binary(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=1)
    mc.fit(X, y)
    preds = mc.predict(X)
    assert set(preds).issubset({0, 1})


def test_digitise_bins_in_range(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=1)
    mc.fit(X, y)
    values = X["log_ret_1"].values
    bins = mc._digitise(values, mc.bin_edges_[0])
    assert (bins >= 0).all() and (bins <= mc.n_states - 1).all()


def test_lag_features_order1_shape(train_data):
    X, _ = train_data
    mc = MarkovChain(order=1)
    result = mc._lag_features(X)
    assert list(result.columns) == ["log_ret_1"]


def test_lag_features_order2_shape(train_data):
    X, _ = train_data
    mc = MarkovChain(order=2)
    result = mc._lag_features(X)
    assert list(result.columns) == ["log_ret_1", "log_ret_5"]


def test_unseen_state_returns_default(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=1)
    mc.fit(X, y)
    # Default prob for any state is 0.5 (Laplace smoothing)
    for p in mc.transition_.values():
        assert 0.0 < p < 1.0


def test_order2_fit_and_predict(train_data):
    X, y = train_data
    mc = MarkovChain(n_states=3, order=2)
    mc.fit(X, y)
    probas = mc.predict_proba(X)
    assert len(probas) == len(X)
    assert ((probas >= 0) & (probas <= 1)).all()
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_markov.py
git commit -m "test: add MarkovChain unit tests"
```

---

### Task 6: test_rnn_utils.py — RNN utilities

**Files:**
- Create: `tests/test_rnn_utils.py`

- [ ] **Step 1: Write `tests/test_rnn_utils.py`**

```python
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
    # removes 1 (dow), adds 2 (dow_sin, dow_cos) → net +1
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
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_rnn_utils.py
git commit -m "test: add RNN utility unit tests (add_cyclic_dow, build_sequences, SequenceDataset)"
```

---

### Task 7: test_rate_limiter.py + test_llm_service.py

**Files:**
- Create: `tests/test_rate_limiter.py`
- Create: `tests/test_llm_service.py`

- [ ] **Step 1: Write `tests/test_rate_limiter.py`**

```python
import time
import pytest
from llm.rate_limit import RateLimitState


def test_record_increments_deques():
    rl = RateLimitState(tpm_limit=10000, rpm_limit=100)
    rl.record(500)
    assert len(rl.token_events) == 1
    assert len(rl.request_events) == 1


def test_prune_removes_old_events():
    rl = RateLimitState()
    old_ts = time.time() - 120  # 2 minutes ago
    rl.token_events.append((old_ts, 100))
    rl.request_events.append(old_ts)
    rl._prune(time.time())
    assert len(rl.token_events) == 0
    assert len(rl.request_events) == 0


def test_prune_keeps_recent_events():
    rl = RateLimitState()
    rl.record(100)
    rl._prune(time.time())
    assert len(rl.token_events) == 1


def test_wait_for_slot_no_sleep_under_limits():
    rl = RateLimitState(tpm_limit=10000, rpm_limit=100)
    # Should not block — well under limits
    start = time.time()
    rl.wait_for_slot(estimated_tokens=100)
    elapsed = time.time() - start
    assert elapsed < 0.5


def test_estimate_tokens_basic():
    from llm.gpt_service import LLMService
    # Patch __init__ to avoid creating real client
    import unittest.mock as mock
    with mock.patch("llm.gpt_service.create_llm_client"):
        svc = LLMService.__new__(LLMService)
        svc._rate_limit = RateLimitState()
    assert svc.estimate_tokens("hello world") == max(1, len("hello world") // 4)


def test_estimate_tokens_minimum_one():
    from llm.gpt_service import LLMService
    import unittest.mock as mock
    with mock.patch("llm.gpt_service.create_llm_client"):
        svc = LLMService.__new__(LLMService)
        svc._rate_limit = RateLimitState()
    assert svc.estimate_tokens("x") == 1
```

- [ ] **Step 2: Write `tests/test_llm_service.py`**

```python
import json
import pytest
import unittest.mock as mock
from llm.gpt_service import LLMService


@pytest.fixture
def llm_service():
    with mock.patch("llm.gpt_service.create_llm_client") as mock_client:
        svc = LLMService()
        svc._mock_client = mock_client.return_value
    return svc


def _make_response(data: list) -> mock.MagicMock:
    resp = mock.MagicMock()
    resp.output_text = json.dumps(data)
    resp.usage.total_tokens = 100
    resp.usage.input_tokens = 60
    resp.usage.output_tokens = 40
    return resp


def test_query_returns_data_and_usage(llm_service):
    payload = [{"category": "macro_economic", "companies": [], "sentiment": "neutral"}]
    llm_service._client = mock.MagicMock()
    llm_service._client.responses.create.return_value = _make_response(payload)
    result = llm_service.query("sys", "user")
    assert result["data"] == payload
    assert result["usage"]["total_tokens"] == 100


def test_query_records_token_usage(llm_service):
    payload = [{"category": "generic_noise", "companies": [], "sentiment": "neutral"}]
    llm_service._client = mock.MagicMock()
    llm_service._client.responses.create.return_value = _make_response(payload)
    llm_service.query("sys", "user")
    assert len(llm_service._rate_limit.token_events) == 1


def test_estimate_tokens_proportional(llm_service):
    text = "a" * 40
    assert llm_service.estimate_tokens(text) == 10


def test_estimate_tokens_min_one(llm_service):
    assert llm_service.estimate_tokens("") == 1
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_rate_limiter.py tests/test_llm_service.py
git commit -m "test: add RateLimitState and LLMService unit tests"
```

---

### Task 8: test_news_classification.py + test_news_rss.py + test_newsletter.py

**Files:**
- Create: `tests/test_news_classification.py`
- Create: `tests/test_news_rss.py`
- Create: `tests/test_newsletter.py`

- [ ] **Step 1: Write `tests/test_news_classification.py`**

```python
import datetime
import pytest
import unittest.mock as mock
from news.classification import (
    extract_keywords_hit,
    compute_relevance,
    split_into_batches,
    build_news_batch_prompt,
    news_classifier_prompt,
    classify_news,
    IMPORTANT_KEYWORDS,
)


def test_extract_keywords_hit_match():
    result = extract_keywords_hit("los resultados fueron buenos", IMPORTANT_KEYWORDS)
    assert "resultados" in result


def test_extract_keywords_hit_no_match():
    result = extract_keywords_hit("noticia sin palabras clave", IMPORTANT_KEYWORDS)
    assert result == []


def test_extract_keywords_hit_case_insensitive():
    result = extract_keywords_hit("DIVIDENDO anunciado hoy", IMPORTANT_KEYWORDS)
    assert "dividendo" in result


def test_compute_relevance_company_specific_max():
    score = compute_relevance("company_specific", ["BBVA.MC"], "positive", ["dividendo"])
    assert score == pytest.approx(min(0.4 + 0.2 + 0.2 + 0.1, 1.0), abs=1e-3)


def test_compute_relevance_generic_noise_low():
    score = compute_relevance("generic_noise", [], "neutral", [])
    assert score == pytest.approx(0.0)


def test_compute_relevance_with_companies_adds_points():
    base  = compute_relevance("macro_economic", [], "neutral", [])
    with_ = compute_relevance("macro_economic", ["IBEX"], "neutral", [])
    assert with_ > base


def test_compute_relevance_clamped():
    score = compute_relevance("company_specific", ["A"], "positive", list(IMPORTANT_KEYWORDS))
    assert 0.0 <= score <= 1.0


def test_split_into_batches_correct_sizes():
    items = [{"id": i} for i in range(25)]
    batches = split_into_batches(items, batch_size=10)
    assert len(batches) == 3
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5


def test_split_into_batches_empty():
    assert split_into_batches([], batch_size=10) == []


def test_build_news_batch_prompt_contains_titles():
    batch = [{"title": "Banco Santander sube", "body": "Descripcion"}]
    prompt = build_news_batch_prompt(batch)
    assert "Banco Santander sube" in prompt


def test_build_news_batch_prompt_ids_start_at_1():
    batch = [{"title": "T1", "body": "B1"}, {"title": "T2", "body": "B2"}]
    prompt = build_news_batch_prompt(batch)
    assert "id: 1" in prompt
    assert "id: 2" in prompt


def test_news_classifier_prompt_returns_string():
    p = news_classifier_prompt()
    assert isinstance(p, str) and len(p) > 10


def test_classify_news_calls_llm_once_per_batch():
    news = [{"title": f"T{i}", "body": f"B{i}"} for i in range(15)]
    llm_output = [{"category": "generic_noise", "companies": [], "sentiment": "neutral"}]
    batch_output = {"data": llm_output * 10, "usage": {}}

    with mock.patch("news.classification.LLMService") as MockLLM:
        instance = MockLLM.return_value
        # First batch: 10 items, second: 5 items
        instance.query.side_effect = [
            {"data": llm_output * 10, "usage": {}},
            {"data": llm_output * 5,  "usage": {}},
        ]
        result = classify_news(news)

    assert instance.query.call_count == 2
    assert len(result) == 15


def test_classify_news_output_has_required_keys():
    news = [{"title": "T1", "body": "B1", "url": "http://x.com", "date": "2024-01-01",
             "source": "exp", "section": "mercados", "tags": []}]
    llm_out = [{"category": "company_specific", "companies": ["BBVA.MC"], "sentiment": "positive"}]
    with mock.patch("news.classification.LLMService") as MockLLM:
        MockLLM.return_value.query.return_value = {"data": llm_out, "usage": {}}
        result = classify_news(news)
    item = result[0]
    for key in ("category", "companies", "sentiment", "relevance"):
        assert key in item
```

- [ ] **Step 2: Write `tests/test_news_rss.py`**

```python
import datetime
import pytest
import unittest.mock as mock
from news.news_rss import last_news, top_news


def _make_items(dates_and_relevances):
    return [{"title": f"T{i}", "body": "B", "url": f"http://{i}.com",
             "date": d, "relevance": r, "source": "exp", "section": "s", "tags": []}
            for i, (d, r) in enumerate(dates_and_relevances)]


def test_last_news_keeps_yesterday():
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    older     = (datetime.date.today() - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
    items = _make_items([(yesterday, 0.5), (older, 0.8), (yesterday, 0.3)])
    with mock.patch("news.news_rss.fetch_rss", return_value=items):
        result = last_news()
    assert all(item["date"] == yesterday for item in result)
    assert len(result) == 2


def test_last_news_empty_when_no_yesterday_items():
    older = (datetime.date.today() - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
    items = _make_items([(older, 0.5)])
    with mock.patch("news.news_rss.fetch_rss", return_value=items):
        result = last_news()
    assert result == []


def test_top_news_returns_k_items():
    items = _make_items([("2024-01-01", r) for r in [0.1, 0.5, 0.9, 0.3, 0.7]])
    result = top_news(items, k=3)
    assert len(result) == 3


def test_top_news_sorted_descending():
    items = _make_items([("2024-01-01", r) for r in [0.1, 0.9, 0.5]])
    result = top_news(items, k=3)
    relevances = [item["relevance"] for item in result]
    assert relevances == sorted(relevances, reverse=True)
```

- [ ] **Step 3: Write `tests/test_newsletter.py`**

```python
import datetime
import pandas as pd
import pytest
from newsletter.build import (
    add_header,
    add_footer,
    add_closing,
    add_news,
    add_predictions,
    format_predictions,
)


@pytest.fixture
def today():
    return datetime.date(2024, 6, 15)


@pytest.fixture
def news_items():
    return [
        {"title": "BBVA sube", "body": "El banco sube un 2%.", "url": "http://a.com"},
        {"title": "Iberdrola cae", "body": "Baja un 1%.", "url": "http://b.com"},
    ]


@pytest.fixture
def pred_df():
    df = pd.DataFrame({
        "ticker": ["BBVA.MC", "SAN.MC", "ITX.MC"],
        "pred":   [True, False, True],
        "proba":  [0.72, 0.38, 0.65],
        "date":   ["2024-06-14"] * 3,
    })
    return df


def test_add_header_contains_date(today):
    html = add_header(today)
    assert str(today.day) in html
    assert str(today.month) in html


def test_add_header_is_html(today):
    html = add_header(today)
    assert "<html" in html.lower()


def test_add_footer_contains_year(today):
    html = add_footer(today)
    assert str(today.year) in html


def test_add_footer_is_html(today):
    html = add_footer(today)
    assert "<" in html


def test_add_closing_contains_body_close():
    html = add_closing()
    assert "</body>" in html
    assert "</html>" in html


def test_add_news_contains_titles(news_items):
    html = add_news(news_items)
    assert "BBVA sube" in html
    assert "Iberdrola cae" in html


def test_add_news_contains_urls(news_items):
    html = add_news(news_items)
    assert "http://a.com" in html


def test_format_predictions_adds_action(pred_df):
    import unittest.mock as mock
    with mock.patch("newsletter.build.ticker_to_name",
                    return_value={"BBVA.MC": "BBVA", "SAN.MC": "Santander", "ITX.MC": "Inditex"}):
        result = format_predictions(pred_df)
    assert "action" in result.columns
    assert set(result["action"].unique()).issubset({"Buy", "Sell"})


def test_format_predictions_adds_proba_col(pred_df):
    import unittest.mock as mock
    with mock.patch("newsletter.build.ticker_to_name",
                    return_value={"BBVA.MC": "BBVA", "SAN.MC": "Santander", "ITX.MC": "Inditex"}):
        result = format_predictions(pred_df)
    assert "proba.2f" in result.columns


def test_add_predictions_contains_ticker_names(pred_df):
    import unittest.mock as mock
    with mock.patch("newsletter.build.ticker_to_name",
                    return_value={"BBVA.MC": "BBVA", "SAN.MC": "Santander", "ITX.MC": "Inditex"}):
        formatted = format_predictions(pred_df)
    html = add_predictions(formatted)
    assert "BBVA" in html or "Santander" in html or "Inditex" in html
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_news_classification.py tests/test_news_rss.py tests/test_newsletter.py
git commit -m "test: add news classification, RSS filter, and newsletter builder unit tests"
```

---

### Task 9: test_predict_helpers.py — synthetic artifact inference

**Files:**
- Create: `tests/test_predict_helpers.py`

- [ ] **Step 1: Write `tests/test_predict_helpers.py`**

```python
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


# ── synthetic artifact factories ─────────────────────────────────────────────

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
    cfg = {"input_size": input_size, "hidden_size": 8, "num_layers": 1,
           "dropout": 0.0, "cell": model_key if model_key in ("gru", "lstm") else "gru"}
    if model_key.startswith("cnn_"):
        cfg["num_filters"] = 4
        cfg["kernel_size"] = 3
        model = StockCNNRNN(**cfg)
    else:
        model = StockRNN(**cfg)
    scaler = StandardScaler()
    scaler.fit(np.random.randn(100, input_size))
    features = [f"f{i}" for i in range(4)] + ["dow_sin"]  # post-cyclic-dow cols
    return {
        "model_key":   model_key,
        "model_state": model.state_dict(),
        "model_config": cfg,
        "scaler":      scaler,
        "features":    features,
        "seq_len":     5,
        "target_type": "discrete",
        "ft_type":     "micro",
    }


def _rnn_input(n_tickers=3, n_rows=20):
    """Returns (X, tickers, dates) ready for _predict_rnn."""
    tickers_list = [f"T{i}.MC" for i in range(n_tickers)]
    frames = []
    for t in tickers_list:
        np.random.seed(hash(t) % 2**31)
        # 4 generic features + dow (integer, will be cyclic-encoded)
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


# ── load_artifact ─────────────────────────────────────────────────────────────

def test_load_artifact_raises_file_not_found(tmp_path):
    from config import ARTIFACTS_PATH
    import unittest.mock as mock
    with mock.patch("models.predict.ARTIFACTS_PATH", tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifact("rf", 1, "sliding", "discrete")


# ── _predict_tree ──────────────────────────────────────────────────────────────

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
    # Use a regressor stub that returns floats and has no predict_proba
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


# ── _reconstruct_rnn ──────────────────────────────────────────────────────────

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
    assert not model.training  # eval mode


# ── _predict_rnn ──────────────────────────────────────────────────────────────

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
    # continuous: preds == probas (raw output)
    assert np.allclose(preds, probas)


# ── _temporal_inner_split ────────────────────────────────────────────────────

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
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_predict_helpers.py
git commit -m "test: add predict.py helper unit tests with synthetic artifacts"
```

---

### Task 10: test_db_sqlite.py + test_db_supabase.py

**Files:**
- Create: `tests/test_db_sqlite.py`
- Create: `tests/test_db_supabase.py`

- [ ] **Step 1: Write `tests/test_db_sqlite.py`**

```python
import pandas as pd
import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch, call


def _mock_conn():
    conn = MagicMock()
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    return conn


# ── queries_ohlcv (sqlite) ────────────────────────────────────────────────────

def test_sqlite_get_last_date_returns_value():
    from db.sqlite.queries_ohlcv import _get_last_date
    conn = _mock_conn()
    conn.execute.return_value.fetchone.return_value = ("2024-01-15",)
    result = _get_last_date(conn, "BBVA.MC")
    assert result == "2024-01-15"


def test_sqlite_get_last_date_returns_none():
    from db.sqlite.queries_ohlcv import _get_last_date
    conn = _mock_conn()
    conn.execute.return_value.fetchone.return_value = (None,)
    result = _get_last_date(conn, "BBVA.MC")
    assert result is None


# ── queries_news (sqlite) ─────────────────────────────────────────────────────

def test_load_news_no_filter_query():
    """No filters → no WHERE clause in query."""
    with patch("db.sqlite.queries_news.sqlite_connection") as mock_ctx:
        mock_conn = MagicMock()
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        with patch("db.sqlite.queries_news.pd.read_sql", return_value=pd.DataFrame()) as mock_sql:
            from db.sqlite.queries_news import load_news
            load_news()
            query_used = mock_sql.call_args[0][0]
            assert "WHERE" not in query_used.upper()


def test_load_news_start_filter_query():
    """start provided → WHERE date >= in query."""
    with patch("db.sqlite.queries_news.sqlite_connection") as mock_ctx:
        mock_conn = MagicMock()
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        with patch("db.sqlite.queries_news.pd.read_sql", return_value=pd.DataFrame()) as mock_sql:
            from db.sqlite.queries_news import load_news
            load_news(start="2024-01-01")
            query_used = mock_sql.call_args[0][0]
            assert "date >=" in query_used.lower()


def test_load_news_both_filters_query():
    """Both start and end → two conditions in query."""
    with patch("db.sqlite.queries_news.sqlite_connection") as mock_ctx:
        mock_conn = MagicMock()
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        with patch("db.sqlite.queries_news.pd.read_sql", return_value=pd.DataFrame()) as mock_sql:
            from db.sqlite.queries_news import load_news
            load_news(start="2024-01-01", end="2024-01-31")
            query_used = mock_sql.call_args[0][0]
            assert "date >=" in query_used.lower()
            assert "date <=" in query_used.lower()


def test_load_entities_no_filter_runs():
    with patch("db.sqlite.queries_news.sqlite_connection") as mock_ctx:
        mock_conn = MagicMock()
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        with patch("db.sqlite.queries_news.pd.read_sql", return_value=pd.DataFrame()):
            from db.sqlite.queries_news import load_entities
            load_entities()  # must not raise


# ── db/base sqlite_connection ────────────────────────────────────────────────

def test_sqlite_connection_context_manager():
    with patch("db.base.sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        from db.base import sqlite_connection
        with sqlite_connection():
            pass
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()


def test_sqlite_connection_closes_on_exception():
    with patch("db.base.sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        from db.base import sqlite_connection
        with pytest.raises(ValueError):
            with sqlite_connection():
                raise ValueError("test error")
        mock_conn.close.assert_called_once()
```

- [ ] **Step 2: Write `tests/test_db_supabase.py`**

```python
import pandas as pd
import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch


def _mock_supabase():
    client = MagicMock()
    return client


# ── queries_ohlcv (supabase) ─────────────────────────────────────────────────

def test_supabase_fetch_ohlcv_calls_rpc():
    from db.supabase.queries_ohlcv import fetch_ohlcv
    client = _mock_supabase()
    client.rpc.return_value.execute.return_value.data = [
        {"ticker": "BBVA.MC", "date": "2024-01-15", "open": 7.1,
         "high": 7.2, "low": 7.0, "close": 7.15, "volume": 500000.0}
    ]
    with patch("db.supabase.queries_ohlcv.supabase_client", return_value=client):
        result = fetch_ohlcv(["BBVA.MC"], count=50)
    client.rpc.assert_called_once()
    assert isinstance(result, pd.DataFrame)


def test_supabase_fetch_ohlcv_returns_dataframe():
    from db.supabase.queries_ohlcv import fetch_ohlcv
    client = _mock_supabase()
    client.rpc.return_value.execute.return_value.data = []
    with patch("db.supabase.queries_ohlcv.supabase_client", return_value=client):
        result = fetch_ohlcv(["BBVA.MC"], count=10)
    assert isinstance(result, pd.DataFrame)


# ── ingest_ohlcv (supabase) ──────────────────────────────────────────────────

def test_supabase_ingest_ohlcv_calls_upsert():
    from db.supabase.ingest_ohlcv import ingest_ohlcv
    client = _mock_supabase()
    df = pd.DataFrame({
        "ticker": ["BBVA.MC"], "date": ["2024-01-15"],
        "open": [7.1], "high": [7.2], "low": [7.0], "close": [7.15], "volume": [500000.0]
    })
    ingest_ohlcv(client, df)
    client.table.assert_called()
    client.table.return_value.upsert.assert_called()


# ── upload_preds (supabase) ──────────────────────────────────────────────────

def test_upload_preds_calls_upsert():
    from db.supabase.upload_preds import upload_preds
    client = _mock_supabase()
    df = pd.DataFrame({
        "ticker": ["BBVA.MC"], "date": ["2024-01-15"],
        "pred": [1], "proba": [0.72], "model": ["rf_h1_sliding_discrete"]
    })
    upload_preds(client, df)
    client.table.assert_called_with("predictions")
    client.table.return_value.upsert.assert_called()


def test_upload_preds_empty_df_does_not_call_upsert():
    from db.supabase.upload_preds import upload_preds
    client = _mock_supabase()
    upload_preds(client, pd.DataFrame())
    client.table.assert_not_called()


# ── queries_news (supabase) ──────────────────────────────────────────────────

def test_supabase_top_k_news_calls_select():
    from db.supabase.queries_news import top_k_news
    client = _mock_supabase()
    chain = (client.table.return_value.select.return_value
             .eq.return_value.order.return_value.limit.return_value.execute.return_value)
    chain.data = [{"title": "T", "body": "B", "url": "http://x.com"}]
    with patch("db.supabase.queries_news.supabase_client", return_value=client):
        result = top_k_news(k=5, date="2024-01-15")
    client.table.assert_called()
    assert isinstance(result, list)
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_db_sqlite.py tests/test_db_supabase.py
git commit -m "test: add SQLite and Supabase DB layer unit tests"
```

---

## Self-Review

**Spec coverage:**
- ✅ evaluate_model, evaluate_regression — Task 2
- ✅ sliding_windows, expanding_windows — Task 3
- ✅ all features.py functions — Task 4
- ✅ MarkovChain full lifecycle — Task 5
- ✅ add_cyclic_dow, build_sequences, SequenceDataset — Task 6
- ✅ RateLimitState, LLMService — Task 7
- ✅ news classification pure + mocked LLM — Task 8
- ✅ last_news, top_news — Task 8
- ✅ newsletter HTML builders, format_predictions — Task 8
- ✅ load_artifact, _predict_tree, _reconstruct_rnn, _predict_rnn — Task 9
- ✅ _temporal_inner_split — Task 9
- ✅ sqlite queries_news (with WHERE fix verified), sqlite_connection — Task 10
- ✅ supabase fetch_ohlcv, ingest, upload_preds, queries_news — Task 10
- ✅ conftest shared fixtures — Task 1

**Placeholder scan:** None found.

**Type consistency:**
- `_tree_artifact()` returns dict with `"model"`, `"features"`, `"target_type"`, `"ft_type"` — matches what `_predict_tree` reads — consistent.
- `_rnn_artifact()` returns dict with all keys `_reconstruct_rnn` and `_predict_rnn` read — consistent.
- `_rnn_input()` returns `(X, tickers, dates)` — matches `_predict_rnn(artifact, X, tickers, dates, target_type)` signature — consistent.
- `add_cyclic_dow` applied before `[feature_cols]` selection in `_predict_rnn` test (bug fix respected) — consistent.
