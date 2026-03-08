import pandas as pd
from supabase import Client

from config import load_env, DATA_PATH
from db.base import supabase_client
from models.predict import get_predictions


def upload_preds(supabase: Client, df: pd.DataFrame) -> None:
    """Enters new ohlcv values in database."""
    if df.empty:
        return None

    records = df.to_dict(orient="records")

    res = (
        supabase
        .table("predictions")
        .upsert(
            records,
            on_conflict="ticker,date,model"
        )
        .execute())

if __name__ == "__main__":
    load_env()

    supabase = supabase_client()

    predictions = get_predictions()

    # Update daily predictions repo
    path = DATA_PATH / "predictions.json"
    predictions[["ticker", "pred", "proba", "date"]].to_json(path, orient="records",index=False)

    # Upload predictions cloud
    upload_preds(supabase, predictions)
