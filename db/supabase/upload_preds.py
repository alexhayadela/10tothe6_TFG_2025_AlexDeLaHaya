import pandas as pd
from supabase import Client

from utils import load_env
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
    upload_preds(supabase, predictions)
