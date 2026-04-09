import joblib
import numpy as np
import pandas as pd

from config import load_env, ARTIFACTS_PATH
from db.supabase.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers
from models.trees.features import ml_ready  # safe_build_features,


def load_model(type: str, horizon: int):
    """Load a trained model artifact from disk.

    Reads a .pkl file named `{type}_h{horizon}_full.pkl` from the artifacts
    directory (e.g. rf_h1_full.pkl). Returns the fitted model object and the
    ordered list of feature column names used during training — the column
    order must be reproduced exactly at inference time.
    """
    name = f"{type}_h{horizon}_full.pkl"
    artifact = joblib.load(ARTIFACTS_PATH / name)
    model = artifact["model"]
    feature_cols = artifact["features"]

    return model, feature_cols


def _get_predictions(model_type: str, horizon: int) -> pd.DataFrame:
    """Run the full inference pipeline and return one prediction per ticker.

    Steps:
    1. Fetch the last 50 OHLCV bars per IBEX35 ticker from Supabase
       (enough history for all rolling indicators).
    2. Build micro features and drop rows with NaN (mask).
    3. Run model.predict and model.predict_proba; `pred_proba` is the
       probability of the predicted class (not always the buy probability).
    4. Keep only the most recent prediction per ticker (last date after sort).

    Returns a DataFrame with columns: ticker, date, pred (0=sell/1=buy), proba.
    """
    tickers = get_ibex_tickers()
    df_micro = fetch_ohlcv(tickers, 50)

    model, feature_cols = load_model(model_type, horizon)

    # MISSING: enforce model needs -> df_macro, type{micro, cross, micro}
    df, X, _, mask = ml_ready(horizon, df_micro, None, "micro")
    df_pred = df.loc[mask, ["ticker", "date"]]

    preds = model.predict(X)
    probas = model.predict_proba(X)
    pred_proba = probas[np.arange(len(preds)), preds]
    #confidence = np.abs(prob_buy - prob_sell)

    df_pred["pred"] = preds
    df_pred["proba"] = pred_proba
    # df_pred["confidence"] = confidence

    df_pred = df_pred.sort_values("date").groupby("ticker").tail(1)
    return df_pred


def get_predictions():
    df = _get_predictions(model_type="rf", horizon=1)
    df["model"] = "rf_1"
    return df


if __name__ == "__main__":
    load_env()

    df = get_predictions()
    print(df)
