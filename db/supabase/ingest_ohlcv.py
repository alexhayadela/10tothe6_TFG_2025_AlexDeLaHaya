import pandas as pd
from supabase import Client

from utils import load_env
from db.base import supabase_client
from db.utils_ohlcv import get_all_tickers, download_ticker
from db.supabase.queries_ohlcv import _get_last_date


def ingest_ohlcv(supabase: Client, df: pd.DataFrame) -> None:
    """Enters new ohlcv values in database."""
    if df.empty:
        return None

    records = df.to_dict(orient="records")

    res = (
        supabase
        .table("ohlcv")
        .upsert(
            records,
            on_conflict="ticker,date"
        )
        .execute())


def update_ticker(supabase: Client, ticker: str) -> pd.DataFrame:
    """Returns updated ticker info."""
    last_date = _get_last_date(supabase, ticker)
    start = None
    if last_date:
        start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    df = download_ticker(ticker, start=start)
    return df


def update_tickers(supabase: Client, tickers: list[str]) -> pd.DataFrame:
    """Returns updated info for a group of tickers."""
    dfs = []
    for t in tickers:
        try:
            df = update_ticker(supabase, t)
            if not df.empty:
                dfs.append(df)
        except Exception: # A ticker may be decomissioned.
            continue
    
    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    load_env()
    
    supabase = supabase_client()
    tickers = get_all_tickers()

    df = update_tickers(supabase, tickers)
    ingest_ohlcv(supabase, df)
