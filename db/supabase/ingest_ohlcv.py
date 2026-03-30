import pandas as pd
from supabase import Client

from config import load_env
from db.base import supabase_client
from db.utils_ohlcv import get_all_tickers, download_ticker
from db.supabase.queries_ohlcv import _get_last_date


def ingest_ohlcv(supabase: Client, df: pd.DataFrame) -> None:
    """Upsert OHLCV rows into Supabase, deduplicating on (ticker, date).

    Converts the DataFrame to a list of dicts and issues an upsert so that
    re-running the ingestion with overlapping dates is safe — existing rows
    are updated instead of raising a unique constraint error.
    """
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
    """Incrementally download missing OHLCV bars for a single ticker.

    Checks the latest stored date in Supabase; if found, fetches only bars
    from the next day onwards to avoid re-downloading existing data.
    If no data exists yet, fetches the full history.
    """
    last_date = _get_last_date(supabase, ticker)
    start = None
    if last_date:
        start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    df = download_ticker(ticker, start=start)
    return df


def update_tickers(supabase: Client, tickers: list[str]) -> pd.DataFrame:
    """Incrementally update a list of tickers and return the combined new rows.

    Calls update_ticker for each ticker, silently skipping failures
    (e.g. delisted symbols). Returns the concatenated DataFrame of all
    new bars, ready to pass to ingest_ohlcv.
    """
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
