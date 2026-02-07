import pandas as pd
import numpy as np


def rolling_slope(series, window):
    x = np.arange(window)
    return series.rolling(window).apply(
        lambda y: np.polyfit(x, y, 1)[0],
        raw=True
    )


def rsi(series, window):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def micro_features(df:pd.DataFrame):
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


def cross_micro_features(df:pd.DataFrame):
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


def target_feature(df:pd.DataFrame, horizon):
    df = df.copy()
    df["future_log_ret"] = np.log(df["close"].shift(-horizon) / df["close"])
    df["target"] = (df["future_log_ret"] > 0).astype(int)

    return df


def assert_columns(df: pd.DataFrame, required):
    missing = set(required) - set(df.columns)
    if missing:
        raise AssertionError(f"Dataframe is missing required columns:{sorted(missing)}")
    

def final_features(df:pd.DataFrame):
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
    cross_micro = ["ibx_breadth", "ibx_breadth_10d"]

    keep_macro = [
        "ibx_vol_10","ibx_vol_ratio_10_60",
        "sp_vol_20","sp_vol_ratio_20_100",
        "vix_chg_z_5","vix_pctile_250",
        "rel_ret_5","rel_ret_20","rel_vol_20"]
    drop_macro = [
    "ibx_log_ret_1","ibx_log_ret_5","ibx_log_ret_20",
    "sp_log_ret_1", "vix_chg_1",
    "ibx_vol_20","ibx_vol_60", "sp_vol_100"]

    assert_columns(df, keep_micro + keep_macro + drop_micro + drop_macro + cross_micro)
    df = df.copy()
    df = df.drop(columns= drop_micro + drop_macro)
    return df


def build_features(df_micro:pd.DataFrame, df_macro:pd.DataFrame, horizon:int) -> pd.DataFrame:
    df_final = []
    
    df_micro = df_micro.sort_values(["ticker", "date"]).reset_index(drop=True)
    df_macro = df_macro.sort_values("date").reset_index(drop=True)
    macro = macro_features(df_macro)
    for ticker, df_t in df_micro.groupby("ticker"):
        
        df_t = micro_features(df_t)
        df_t = df_t.merge(macro, on="date", how="left")  
        df_t = target_feature(df_t,horizon)
        df_t["ticker"] = ticker
        
        df_final.append(df_t)
    df_final = pd.concat(df_final, ignore_index=True)
    df_final = cross_micro_features(df_final)
    #df_final = final_features(df_final)
    
    return df_final

# FIX TO NANS, compute macro outside

def macro_features(df_macro: pd.DataFrame):

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

    # IBEX
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
"""
df["rel_ret_5"] = df["log_ret_5"] - df["ibx_log_ret_5"]
    df["rel_ret_20"] = df["log_ret_20"] - df["ibx_log_ret_20"]
    df["rel_vol_20"] = df["vol_20"] / df["ibx_vol_20"]
"""


