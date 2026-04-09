import numpy as np
import pandas as pd
from typing import Literal



# ── helpers ───────────────────────────────────────────────────────────────────

def assert_columns(df: pd.DataFrame, required):
    missing = set(required) - set(df.columns)
    if missing:
        raise AssertionError(f"DataFrame missing columns: {sorted(missing)}")


def rolling_slope(series, window):
    """Compute the linear trend slope over a rolling window.

    Fits a degree-1 polynomial (y = a*x + b) to each window using least
    squares; returns the slope coefficient `a`. Positive = uptrend,
    negative = downtrend. Magnitude reflects steepness.
    """
    x = np.arange(window)
    return series.rolling(window).apply(
        lambda y: np.polyfit(x, y, 1)[0], raw=True
    )


def rsi(series, window):
    """Compute the Relative Strength Index over a rolling window.

    Separates daily price changes into gains and losses, takes their rolling
    averages, and scales the ratio to 0-100. >70 = overbought, <30 = oversold.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── per-stock features ────────────────────────────────────────────────────────

def micro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all single-stock technical indicators from raw OHLCV data.

    Expects a single-ticker DataFrame sorted by date with columns:
    open, high, low, close, volume. Adds columns covering:
    - Log returns: 1, 5, 10, 20 days
    - Rolling volatility (std of log_ret_1): 5, 10, 20 days + ratio 5/20
    - ATR(14) as % of close
    - SMA 5/10/20/50 and EMA 5/10/20/50 (raw dropped later; ratios kept)
    - MACD histogram normalised by close: (EMA12 - EMA26 - signal9) / close
    - Bollinger %B (20-day, 2σ): position of close within the bands
    - Rolling slope of close over 10 days
    - Distance to 10/20-day rolling highs and lows
    - RSI(14)
    - Volume ratios (5-day, 20-day) and volume-weighted return
    - OBV slope (10-day) normalised by 20-day avg volume
    - Amihud illiquidity: 10-day rolling mean of |log_ret_1| / volume
    - Return autocorrelation lag-1 over a 10-day rolling window
    - Candlestick: intraday return, body, upper/lower wicks, gap
    Does NOT add the target variable — call target_feature separately.
    """
    assert_columns(df, ["open", "high", "low", "close", "volume"])
    df = df.copy()

    # Returns
    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    for w in [5, 10, 20]:
        df[f"log_ret_{w}"] = np.log(df["close"] / df["close"].shift(w))

    # Volatility
    for w in [5, 10, 20]:
        df[f"vol_{w}"] = df["log_ret_1"].rolling(w).std()
    df["vol_ratio_5_20"] = df["vol_5"] / df["vol_20"]

    # ATR(14)
    high_low   = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift(1))
    low_close  = np.abs(df["low"]  - df["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    # Moving averages
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    df["sma_ratio_5_20"]  = df["sma_5"]  / df["sma_20"]  - 1
    df["sma_ratio_10_50"] = df["sma_10"] / df["sma_50"]  - 1

    # MACD histogram: (EMA12 - EMA26 - signal9) / close
    # Captures momentum divergence; normalised to be scale-invariant across tickers.
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line   = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd_line - signal_line) / df["close"]

    # Bollinger %B (20-day, 2σ)
    # 0 = at lower band (oversold), 1 = at upper band (overbought), 0.5 = at mid.
    bb_std   = df["close"].rolling(20).std()
    bb_upper = df["sma_20"] + 2 * bb_std
    bb_lower = df["sma_20"] - 2 * bb_std
    df["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)

    # Trend slope (10-day only)
    df["slope_10"] = rolling_slope(df["close"], 10)

    # Distance to rolling highs/lows — mean-reversion signal
    for w in [10, 20]:
        df[f"dist_high_{w}"] = df["close"] / df["high"].rolling(w).max() - 1
        df[f"dist_low_{w}"]  = df["close"] / df["low"].rolling(w).min()  - 1

    # RSI(14)
    df["rsi_14"] = rsi(df["close"], 14)

    # Volume features
    for w in [5, 20]:
        df[f"volu_mean_{w}"] = df["volume"].rolling(w).mean()
        df[f"volu_ratio_{w}"] = df["volume"] / df[f"volu_mean_{w}"]
    df["volu_ret_1"] = df["log_ret_1"] * df["volu_ratio_5"]

    # OBV slope: direction-weighted cumulative volume trend.
    # Normalised by 20-day avg volume so it's comparable across tickers.
    obv = (df["volume"] * np.sign(df["log_ret_1"])).cumsum()
    df["obv_slope_10"] = rolling_slope(obv, 10) / df["volu_mean_20"].replace(0, np.nan)

    # Amihud (2002) illiquidity: price impact per unit of dollar volume.
    # High values → price moves a lot per unit traded (illiquid, fragile).
    amihud_raw = df["log_ret_1"].abs() / df["volume"].replace(0, np.nan)
    df["amihud_10"] = amihud_raw.rolling(10).mean()

    # Lag-1 return autocorrelation over 10-day window.
    # Positive = momentum microstructure; negative = mean reversion.
    df["ret_autocorr_10"] = df["log_ret_1"].rolling(10).corr(df["log_ret_1"].shift(1))

    # Candlestick features
    df["intraday_ret"] = (df["close"] - df["open"]) / df["open"]
    df["body"]        = (df["close"] - df["open"]).abs() / df["open"]
    df["upper_wick"]  = (df["high"] - df[["close", "open"]].max(axis=1)) / df["open"]
    df["lower_wick"]  = (df[["close", "open"]].min(axis=1) - df["low"]) / df["open"]
    df["gap"]         = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    return df


def horizon_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add features that are specifically motivated for a given prediction horizon.

    h=1: day-of-week (0=Mon … 4=Fri) — captures Monday/Friday microstructure effects.
    h=5: month-of-year — captures seasonal/calendar patterns over a multi-day hold.
         12-1 month momentum (Jegadeesh & Titman 1993) — intermediate-term trend signal
         computed as log(close[t-21] / close[t-252]), i.e. the past year return
         excluding the most recent month to avoid short-term reversal contamination.
    Must be called inside the per-ticker loop (after micro_features) since it uses
    the close series and the date column.
    """
    df = df.copy()
    dates = pd.to_datetime(df["date"])

    if horizon == 1:
        df["dow"] = dates.dt.dayofweek

    if horizon == 5:
        df["month"]    = dates.dt.month
        df["mom_12_1"] = np.log(df["close"].shift(21) / df["close"].shift(252))

    return df


# ── cross-stock features ──────────────────────────────────────────────────────

def cross_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-breadth features using leave-one-out calculation.

    For each date, ibx_breadth = fraction of *other* IBEX stocks with a positive
    1-day return (leave-one-out to avoid the stock contributing to its own signal).
    ibx_breadth_10d is the 10-day rolling mean of that value, computed per ticker
    to respect the time ordering within each stock's series.
    Requires a multi-ticker DataFrame with log_ret_1 already computed.
    """
    df = df.copy()

    # Leave-one-out: subtract the current stock's own contribution
    n_up    = df.groupby("date")["log_ret_1"].transform(lambda x: (x > 0).sum())
    n_total = df.groupby("date")["log_ret_1"].transform("count")
    self_up = (df["log_ret_1"] > 0).astype(int)

    df["ibx_breadth"] = (n_up - self_up) / (n_total - 1)

    # Rolling mean per ticker (time-ordered within each stock)
    df["ibx_breadth_10d"] = (
        df.groupby("ticker")["ibx_breadth"]
          .transform(lambda x: x.rolling(10).mean())
    )

    return df


# ── macro / market-level features ────────────────────────────────────────────

def macro_features(df_macro: pd.DataFrame) -> pd.DataFrame:
    """Compute market-level risk and volatility features from index data.

    Expects a long-format DataFrame with tickers ^IBEX, ^GSPC, ^VIX.
    Pivots to wide format then computes:
    - IBEX log returns (1, 5, 20 day) and rolling vol at 10/60 days + ratio
    - S&P 500 log returns and rolling vol at 20/100 days + ratio
    - VIX daily % change, z-scored over 5 days, and 250-day percentile
      (where VIX sits relative to the past year — a risk-regime indicator)
    Returns a flat DataFrame indexed by date for merging with stock data.
    """
    assert_columns(df_macro, ["date", "ticker", "close"])

    macro = (
        df_macro.set_index(["date", "ticker"])["close"]
        .unstack("ticker")
        .rename(columns={"^IBEX": "ibx_close", "^GSPC": "sp_close", "^VIX": "vix_close"})
        .sort_index()
    )

    macro["ibx_log_ret_1"]  = np.log(macro["ibx_close"] / macro["ibx_close"].shift(1))
    macro["ibx_log_ret_5"]  = np.log(macro["ibx_close"] / macro["ibx_close"].shift(5))
    macro["ibx_log_ret_20"] = np.log(macro["ibx_close"] / macro["ibx_close"].shift(20))
    macro["ibx_vol_10"]     = macro["ibx_log_ret_1"].rolling(10).std()
    macro["ibx_vol_20"]     = macro["ibx_log_ret_1"].rolling(20).std()
    macro["ibx_vol_60"]     = macro["ibx_log_ret_1"].rolling(60).std()
    macro["ibx_vol_ratio_10_60"] = macro["ibx_vol_10"] / macro["ibx_vol_60"]

    macro["sp_log_ret_1"]  = np.log(macro["sp_close"] / macro["sp_close"].shift(1))
    macro["sp_vol_20"]     = macro["sp_log_ret_1"].rolling(20).std()
    macro["sp_vol_100"]    = macro["sp_log_ret_1"].rolling(100).std()
    macro["sp_vol_ratio_20_100"] = macro["sp_vol_20"] / macro["sp_vol_100"]

    macro["vix_chg_1"]     = macro["vix_close"].pct_change()
    macro["vix_chg_z_5"]   = macro["vix_chg_1"] / macro["vix_chg_1"].rolling(5).std()
    macro["vix_pctile_250"] = (
        macro["vix_close"].rolling(250).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    )

    return macro.reset_index()


def rel_to_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock return and volatility relative to the IBEX index.

    Call after merging macro features so ibx_log_ret_5/20 and ibx_vol_20 exist.
    rel_ret measures alpha vs. the index; rel_vol measures how volatile the stock
    is relative to the market — both are important for stock selection models.
    """
    df = df.copy()
    df["rel_ret_5"]  = df["log_ret_5"]  - df["ibx_log_ret_5"]
    df["rel_ret_20"] = df["log_ret_20"] - df["ibx_log_ret_20"]
    df["rel_vol_20"] = df["vol_20"]     / df["ibx_vol_20"]
    return df


# ── target ────────────────────────────────────────────────────────────────────

def target_feature(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create the ML classification target for a given prediction horizon.

    Computes future_log_ret = log(close[t+horizon] / close[t]) using
    shift(-horizon) to look forward. Binarises to `target`: 1 if the stock
    will be higher in `horizon` days, 0 otherwise.
    The last `horizon` rows will have NaN targets — excluded via the notna
    mask in ml_ready. Do not call this before all price-based features are
    computed or you risk leakage through look-ahead in rolling calculations.
    """
    df = df.copy()
    df["future_log_ret"] = np.log(df["close"].shift(-horizon) / df["close"])
    df["target"] = (df["future_log_ret"] > 0).astype(int)
    return df


# ── feature selection ─────────────────────────────────────────────────────────

def necessary_features(df: pd.DataFrame,
                        ft_type: Literal["micro", "cross", "macro"] = "macro",
                        horizon: int = 1) -> pd.DataFrame:
    """Drop intermediate/raw columns, keeping only the final ML feature set.

    ft_type controls which feature groups are included:
    - "micro":  single-stock indicators only
    - "cross":  micro + IBEX breadth (leave-one-out)
    - "macro":  cross + macro risk + relative-to-market features

    horizon controls which horizon-specific columns are kept:
    - 1:  dow
    - 5:  month, mom_12_1

    Intermediate columns (raw SMA/EMA/ATR values) are dropped because the
    model only uses their normalised ratios.
    """
    keep_micro = [
        "log_ret_1", "log_ret_5", "log_ret_10", "log_ret_20",
        "vol_5", "vol_ratio_5_20",
        "atr_pct",
        "sma_ratio_5_20", "sma_ratio_10_50",
        "macd_hist", "bb_pct",
        "slope_10",
        "dist_high_10", "dist_low_10", "dist_high_20", "dist_low_20",
        "rsi_14",
        "volu_ratio_5", "volu_ratio_20", "volu_ret_1",
        "obv_slope_10", "amihud_10", "ret_autocorr_10",
        "intraday_ret", "body", "upper_wick", "lower_wick", "gap",
    ]
    drop_micro = [
        "atr_14",
        "vol_10", "vol_20",
        "volu_mean_5", "volu_mean_20",
        "sma_5", "sma_10", "sma_20", "sma_50",
        "ema_5", "ema_10", "ema_20", "ema_50",
    ]

    if horizon == 1:
        keep_horizon = ["dow"]
    elif horizon == 5:
        keep_horizon = ["month", "mom_12_1"]
    else:
        keep_horizon = []

    if ft_type in {"cross", "macro"}:
        keep_cross = ["ibx_breadth", "ibx_breadth_10d"]
    else:
        keep_cross = []

    if ft_type == "macro":
        keep_macro = [
            "ibx_vol_10", "ibx_vol_ratio_10_60",
            "sp_vol_20", "sp_vol_ratio_20_100",
            "vix_chg_z_5", "vix_pctile_250",
            "rel_ret_5", "rel_ret_20", "rel_vol_20",
        ]
        drop_macro = [
            "ibx_log_ret_1", "ibx_log_ret_5", "ibx_log_ret_20",
            "sp_log_ret_1", "vix_chg_1",
            "ibx_vol_20", "ibx_vol_60", "sp_vol_100",
        ]
    else:
        keep_macro = []
        drop_macro = []

    assert_columns(df, keep_micro + drop_micro + keep_horizon + keep_cross + keep_macro + drop_macro)
    df = df.copy()
    df = df.drop(columns=drop_micro + drop_macro)
    return df


# ── pipeline ──────────────────────────────────────────────────────────────────

def build_features(horizon: int,
                   df_micro: pd.DataFrame,
                   df_macro: pd.DataFrame | None = None,
                   ft_type: Literal["micro", "cross", "macro"] = "macro") -> pd.DataFrame:
    """Orchestrate the full feature engineering pipeline for all tickers.

    Per-ticker loop:
    1. micro_features   — technical indicators
    2. horizon_features — horizon-specific features (dow / month / mom_12_1)
    3. target_feature   — forward return label
    4. macro merge      — join macro features by date (ft_type == "macro" only)

    After the loop:
    5. cross_micro_features — IBEX breadth (leave-one-out) for ft_type in {cross, macro}
    6. rel_to_market_features — stock vs. index features (ft_type == "macro" only)
    7. necessary_features — drop intermediate columns, select final feature set

    Returns a flat DataFrame with all tickers/dates and the final feature set
    including `target` (not yet split into X/y — use ml_ready for that).
    """
    df_micro = df_micro.sort_values(["ticker", "date"]).reset_index(drop=True)

    if ft_type == "macro":
        assert df_macro is not None, "df_macro required for ft_type='macro'"
        df_macro = df_macro.sort_values("date").reset_index(drop=True)
        macro = macro_features(df_macro)

    df_final = []
    for ticker, df_t in df_micro.groupby("ticker"):
        df_t = micro_features(df_t)
        df_t = horizon_features(df_t, horizon)
        df_t = target_feature(df_t, horizon)
        df_t["ticker"] = ticker

        if ft_type == "macro":
            df_t = df_t.merge(macro, on="date", how="left")

        df_final.append(df_t)

    df_final = pd.concat(df_final, ignore_index=True)

    if ft_type in {"cross", "macro"}:
        df_final = cross_micro_features(df_final)

    if ft_type == "macro":
        df_final = rel_to_market_features(df_final)

    df_final = necessary_features(df_final, ft_type, horizon)
    return df_final


def ml_ready(horizon: int,
             df_micro: pd.DataFrame,
             df_macro: pd.DataFrame | None = None,
             ft_type: Literal["micro", "cross", "macro"] = "macro"):
    """Build features and prepare arrays for sklearn/PyTorch training or inference.

    Calls build_features, then:
    1. Drops metadata columns (ticker, date, OHLCV, target, future_log_ret)
       to produce a pure numeric feature matrix X.
    2. Replaces ±inf with NaN (can appear in ratio features on zero denominators).
    3. Computes `mask` — rows where ALL features are non-NaN. Early rows are
       NaN-heavy from long rolling windows; the last `horizon` rows have NaN
       targets. Only masked rows are safe to use.

    Returns:
      df     — full DataFrame with metadata + target (for alignment/debugging)
      X      — feature matrix, masked rows only, no NaN
      y      — target Series, masked rows only
      mask   — boolean index to align predictions back to df (ticker/date lookup)
    """
    df = build_features(horizon, df_micro, df_macro, ft_type)
    df = df.sort_values("date").reset_index(drop=True)

    remove_cols = [
        "ticker", "date", "open", "high", "low", "close",
        "volume", "target", "future_log_ret",
    ]

    X = df.drop(columns=remove_cols)
    X = X.replace([np.inf, -np.inf], np.nan)

    mask = X.notna().all(axis=1)

    X = X.loc[mask]
    y = df.loc[mask, "target"]
    return df, X, y, mask
