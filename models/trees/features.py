import datetime
import numpy as np
import pandas as pd
from typing import Literal

# Helper functions
def  assert_columns(df: pd.DataFrame, required):
    missing = set(required) - set(df.columns)
    if missing:
        raise AssertionError(f"Dataframe is missing required columns:{sorted(missing)}")
  
def rolling_slope(series, window):
    """Compute the linear trend slope over a rolling window.

    Fits a degree-1 polynomial (y = a*x + b) to each window using least
    squares; returns the slope coefficient `a`. A positive slope means price
    is trending up over the window; negative means down. Magnitude reflects
    how steep the trend is.
    """
    x = np.arange(window)
    return series.rolling(window).apply(
        lambda y: np.polyfit(x, y, 1)[0],
        raw=True
    )

def rsi(series, window):
    """Compute the Relative Strength Index over a rolling window.

    Separates daily price changes into gains and losses, takes their rolling
    averages, and scales the ratio to 0-100. Values above 70 indicate
    overbought conditions; below 30 indicate oversold.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def micro_features(df: pd.DataFrame):
    """Compute all single-stock technical indicators from raw OHLCV data.

    Expects a single-ticker DataFrame sorted by date with columns:
    open, high, low, close, volume. Adds ~30 new columns covering:
    - Log returns at multiple horizons (1, 3, 5, 10, 20 days)
    - Rolling volatility (std of log returns) and volatility ratios
    - ATR (Average True Range) as a % of close — measures bar-level risk
    - Simple and exponential moving averages (5/10/20/50) and their ratios
    - Rolling price slope (linear trend over 10/20 days)
    - Distance to recent highs/lows (10/20 day) — measures mean-reversion potential
    - RSI(14) and Stochastic K(14) — momentum oscillators
    - Volume ratios and volume-weighted return (volume × log_ret)
    - Candlestick pattern metrics: body size, upper/lower wicks, overnight gap
    Does NOT add the target variable — call target_feature separately.
    """
    assert_columns(df, ["open", "high", "low", "close", "volume"])
    df = df.copy()

    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    for w in [3, 5, 10, 20]:
        df[f"log_ret_{w}"] = np.log(df["close"] / df["close"].shift(w))   
    df["ret_mean_5"] = df["log_ret_1"].rolling(5).mean()
   
    for w in [5, 10, 20]:
        df[f"vol_{w}"] = df["log_ret_1"].rolling(w).std()
    df["vol_ratio_5_20"] = df["vol_5"] / df["vol_20"]

    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift(1))
    low_close = np.abs(df["low"] - df["close"].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    df["sma_ratio_5_20"] = df["sma_5"] / df["sma_20"] - 1
    df["sma_ratio_10_50"] = df["sma_10"] / df["sma_50"] - 1
    df["ema_ratio_5_20"] = df["ema_5"] / df["ema_20"] - 1

    for w in [10, 20]:
        df[f"slope_{w}"] = rolling_slope(df["close"], w)

    for w in [10, 20]:
        df[f"dist_high_{w}"] = df["close"] / df["high"].rolling(w).max() - 1
        df[f"dist_low_{w}"] = df["close"] / df["low"].rolling(w).min() - 1

    df["rsi_14"] = rsi(df["close"], 14)

    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)

    for w in [5, 20]:
        df[f"volu_mean_{w}"] = df["volume"].rolling(w).mean()
        df[f"volu_ratio_{w}"] = df["volume"] / df[f"volu_mean_{w}"]
    df["volu_ret_1"] = df["log_ret_1"] * df["volu_ratio_5"]

    df["body"] = (df["close"] - df["open"]).abs() / df["open"]
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["open"]
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["open"]
    df["true_range_pct"] = true_range / df["close"]
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    return df


def cross_micro_features(df: pd.DataFrame):
    """Add market-breadth features derived from all tickers together.

    Requires a multi-ticker DataFrame (all IBEX stocks). For each date,
    computes `ibx_breadth` = fraction of IBEX stocks with a positive 1-day
    log return. Also computes a 10-day rolling mean of breadth.
    Breadth close to 1 means the whole market is rising (broad rally);
    close to 0 means broad sell-off. Must be called after micro_features
    so that log_ret_1 exists.
    """
    df = df.copy()
    breadth = (
        df.groupby("date")["log_ret_1"]
            .apply(lambda x: (x > 0).mean())
            .rename("ibx_breadth"))
    df = df.merge(
        breadth.reset_index(),
        on="date",
        how="left")
    df["ibx_breadth_10d"] = df["ibx_breadth"].rolling(10).mean()

    return df


def macro_features(df_macro: pd.DataFrame):
    """Compute market-level risk and volatility features from index data.

    Expects a long-format DataFrame with tickers ^IBEX, ^GSPC, ^VIX.
    Pivots to wide format (one column per index) then computes:
    - IBEX log returns and rolling vol at 10/60 days + their ratio
    - S&P 500 log returns and rolling vol at 20/100 days + their ratio
    - VIX daily % change, z-scored over 5 days, and 250-day percentile
      (where VIX sits relative to the past year — a risk regime indicator)
    Returns a flat DataFrame indexed by date, merged with IBEX stock data.
    """
    assert_columns(df_macro, ["date", "ticker", "close"])

    macro = (
        df_macro
        .set_index(["date", "ticker"])["close"]
        .unstack("ticker")
        .rename(columns={
            "^IBEX": "ibx_close",
            "^GSPC": "sp_close",
            "^VIX": "vix_close",
        })
        .sort_index()
    )

    macro["ibx_log_ret_1"] = np.log(macro["ibx_close"] / macro["ibx_close"].shift(1))
    macro["ibx_vol_10"] = macro["ibx_log_ret_1"].rolling(10).std()
    macro["ibx_vol_60"] = macro["ibx_log_ret_1"].rolling(60).std()
    macro["ibx_vol_ratio_10_60"] = macro["ibx_vol_10"] / macro["ibx_vol_60"]

    macro["sp_log_ret_1"] = np.log(macro["sp_close"] / macro["sp_close"].shift(1))
    macro["sp_vol_20"] = macro["sp_log_ret_1"].rolling(20).std()
    macro["sp_vol_100"] = macro["sp_log_ret_1"].rolling(100).std()
    macro["sp_vol_ratio_20_100"] = macro["sp_vol_20"] / macro["sp_vol_100"]

    macro["vix_chg_1"] = macro["vix_close"].pct_change()
    macro["vix_chg_z_5"] = macro["vix_chg_1"] / macro["vix_chg_1"].rolling(5).std()
    macro["vix_pctile_250"] = (
        macro["vix_close"]
        .rolling(250)
        .apply(lambda x: (x <= x[-1]).mean(), raw=True)
    )
    return macro.reset_index()


def rel_to_market_features(df: pd.DataFrame):
    """Compute relative to market (ibex) features.
     
       Call after merging micro/macro."""
    df = df.copy()
    df["rel_ret_5"] = df["log_ret_5"] - df["ibx_log_ret_5"]
    df["rel_ret_20"] = df["log_ret_20"] - df["ibx_log_ret_20"]
    df["rel_vol_20"] = df["vol_20"] / df["ibx_vol_20"]

    return df


def target_feature(df: pd.DataFrame, horizon: int):
    """Create the ML classification target for a given prediction horizon.

    Computes future_log_ret = log(close[t+horizon] / close[t]) using
    shift(-horizon) to look forward. Binarises to `target`: 1 if the stock
    will be higher in `horizon` days, 0 otherwise (buy vs. sell signal).
    The last `horizon` rows will have NaN targets and must be excluded from
    training (ml_ready handles this via the notna mask).
    """
    df = df.copy()
    df["future_log_ret"] = np.log(df["close"].shift(-horizon) / df["close"])
    df["target"] = (df["future_log_ret"] > 0).astype(int)

    return df


def necessary_features(df: pd.DataFrame, ft_type: Literal["micro", "cross", "macro"] = "macro"):
    """Drop intermediate/raw columns, keeping only the final ML feature set.

    `ft_type` controls which groups are kept:
    - "micro": single-stock indicators only (28 features)
    - "cross": micro + IBEX breadth (30 features)
    - "macro": cross + macro risk features + relative-to-market features (39 features)
    Intermediate columns (raw SMA/EMA/ATR values) are dropped because the model
    only needs their normalised ratios, not the raw levels.
    """
    keep_micro = [
    "log_ret_1","log_ret_3","log_ret_5","log_ret_10","log_ret_20","ret_mean_5",
    "slope_10","slope_20","sma_ratio_5_20", "sma_ratio_10_50","ema_ratio_5_20",
    "vol_5","vol_ratio_5_20",
    "atr_pct","true_range_pct","dist_high_10","dist_low_10","dist_high_20","dist_low_20",
    "rsi_14","stoch_k",
    "volu_ratio_5","volu_ratio_20","volu_ret_1",
    "body","upper_wick","lower_wick","gap"]
    drop_micro = [
    "atr_14","vol_10","vol_20","volu_mean_5","volu_mean_20",
    "sma_5","sma_10","sma_20","sma_50",
    "ema_5","ema_10","ema_20","ema_50"]

    if ft_type in {"cross", "macro"}:
        cross_micro = ["ibx_breadth", "ibx_breadth_10d"]
    else: cross_micro = []

    if ft_type == "macro":
        keep_macro = [
            "ibx_vol_10","ibx_vol_ratio_10_60",
            "sp_vol_20","sp_vol_ratio_20_100",
            "vix_chg_z_5","vix_pctile_250",
            "rel_ret_5","rel_ret_20","rel_vol_20"]
        drop_macro = [
        "ibx_log_ret_1","ibx_log_ret_5","ibx_log_ret_20",
        "sp_log_ret_1", "vix_chg_1",
        "ibx_vol_20","ibx_vol_60", "sp_vol_100"]
    else:
        keep_macro = []
        drop_macro = []

    assert_columns(df, keep_micro + keep_macro + drop_micro + drop_macro + cross_micro)
    df = df.copy()
    df = df.drop(columns= drop_micro + drop_macro)
    return df


def build_features(horizon: int, df_micro: pd.DataFrame, df_macro: pd.DataFrame | None = None,
                   ft_type: Literal["micro", "cross", "macro"] = "macro") -> pd.DataFrame:
    """Orchestrate the full feature engineering pipeline for all tickers.

    For each ticker in df_micro (grouped):
    1. Compute micro_features (technical indicators per stock).
    2. Compute target_feature (forward return label for the given horizon).
    3. If ft_type == "macro": merge pre-computed macro features by date.

    After the per-ticker loop:
    4. If ft_type in {"cross", "macro"}: add IBEX breadth via cross_micro_features.
    5. If ft_type == "macro": add relative-to-market features.
    6. Call necessary_features to drop intermediate columns.

    Returns a flat DataFrame with all tickers, all dates, and the final feature
    set including the `target` column (not yet split into X/y).
    """
    df_final = []
    
    df_micro = df_micro.sort_values(["ticker", "date"]).reset_index(drop=True)

    if ft_type == "macro":
        df_macro = df_macro.sort_values("date").reset_index(drop=True)
        macro = macro_features(df_macro)
        macro = align_macro(macro) 

    for ticker, df_t in df_micro.groupby("ticker"):
        
        df_t = micro_features(df_t)
        df_t = target_feature(df_t,horizon)
        df_t["ticker"] = ticker

        if ft_type == "macro":
            df_t = df_t.merge(macro, on="date", how="left")  
            
        df_final.append(df_t)
    df_final = pd.concat(df_final, ignore_index=True)

    if ft_type in {"cross", "macro"}:
        df_final = cross_micro_features(df_final)

    if ft_type == "macro":
        df_final = rel_to_market_features(df_final)
        
    df_final = necessary_features(df_final,ft_type)
    return df_final


def align_macro(df_macro: pd.DataFrame):
    """Forward-fill macro index data to cover all calendar days.

    IBEX and US markets have different holiday calendars, so on days when
    one market is closed there are gaps in the macro data. This function
    reindexes to a complete daily range (from the earliest date to today)
    for each macro ticker and forward-fills missing rows so that every IBEX
    trading day can find a macro value when merging.
    """
    df = df_macro.copy()

    #start_date = "2006-01-01"
    start_date = df["date"].min()
    end_date = datetime.date.today()
    idx_daily = pd.date_range(start=start_date, end=end_date)

    full_index = pd.MultiIndex.from_product(
        [df["ticker"].unique(), idx_daily],
        names=["ticker", "date"]
    )

    df = (
        df
        .set_index(["ticker", "date"])
        .reindex(full_index)
        .groupby(level=0)
        .ffill()
        .reset_index()
    )

    return df

# postprocess features???
def ml_ready(horizon: int, df_micro: pd.DataFrame, df_macro: pd.DataFrame | None = None,
             ft_type: Literal["micro", "cross", "macro"] = "macro"):
    """Build features and prepare arrays for sklearn training or inference.

    Calls build_features, then:
    1. Drops metadata columns (ticker, date, OHLCV, target, future_log_ret)
       to produce a pure numeric feature matrix X.
    2. Replaces inf values with NaN (can appear in ratio features on zero denominators).
    3. Computes `mask` — a boolean Series marking rows where ALL features are
       non-NaN. Early rows are NaN-heavy due to long rolling windows; target rows
       at the end are NaN for the label. Only masked rows are usable.

    Returns:
    - df: full DataFrame including metadata and target (for alignment/debugging)
    - X: feature matrix (only masked rows, no NaN)
    - y: target Series (only masked rows)
    - mask: boolean index to align predictions back to df for ticker/date lookup
    """
    df = build_features(horizon, df_micro, df_macro, ft_type)
    df = df.sort_values("date").reset_index(drop=True)

    remove_cols = [
        "ticker","date","open","high","low","close",
        "volume","target","future_log_ret"
    ]

    X = df.drop(columns=remove_cols)
    #X = X[feature_cols]              # enforce same columns as training
    X = X.replace([np.inf, -np.inf], np.nan)

    mask = X.notna().all(axis=1)

    X = X.loc[mask]
    y = df.loc[mask, "target"]
    #use values for ml train or not
    return df, X, y, mask
