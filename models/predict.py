import joblib
import numpy as np
import pandas as pd

from utils import load_env
from models.utils import get_artifacts_path
from db.supabase.queries_ohlcv import fetch_ohlcv
from db.utils_ohlcv import get_ibex_tickers
from models.trees.features import safe_readyy # safe_build_features,


def load_model(type:str, horizon:int):
    name = f"{type}_h{horizon}_full.pkl"
    artifact = joblib.load(get_artifacts_path() / name)
    model = artifact["model"]
    feature_cols = artifact["features"]
    
    return model, feature_cols


def _get_predictions(model_type:str, horizon:int) -> pd.DataFrame:
    tickers = get_ibex_tickers()
    df_micro = fetch_ohlcv(tickers, 50)

    model, feature_cols = load_model(model_type, horizon)

    df, X, _, mask = safe_readyy(df_micro, horizon, feature_cols)
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