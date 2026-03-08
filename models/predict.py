import joblib
import numpy as np
import pandas as pd

from config import load_env, ARTIFACTS_PATH
from db.supabase.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers
from models.trees.features import ml_ready # safe_build_features,


def load_model(type:str, horizon:int):
    name = f"{type}_h{horizon}_full.pkl"
    artifact = joblib.load(ARTIFACTS_PATH / name)
    model = artifact["model"]
    feature_cols = artifact["features"]
    
    return model, feature_cols


def _get_predictions(model_type:str, horizon:int) -> pd.DataFrame:
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
    #df_pred["confidence"] = confidence

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