import pandas as pd
from supabase import Client
from typing import List
from ingest.ohlcv.utils import get_all_tickers
from ingest.base import supabase_client
from ingest.ohlcv.ingest_sqlite import init_db, get_last_date, append_ohlcv


def fetch_ohlcv_from_supabase(
    supabase: Client,
    ticker: str,
    start_date: str | None = None,
) -> pd.DataFrame:
    query = (
        supabase
        .table("ohlcv")
        .select("ticker,date,open,high,low,close,volume")
        .eq("ticker", ticker)
        .order("date", desc=False)
    )

    if start_date:
        query = query.gt("date", start_date)

    res = query.execute()

    if not res.data:
        return pd.DataFrame()

    return pd.DataFrame(res.data)


def sync_ticker_from_supabase(
    supabase: Client,
    ticker: str,
) -> int:
    last_date = get_last_date(ticker)

    df = fetch_ohlcv_from_supabase(
        supabase,
        ticker=ticker,
        start_date=last_date,
    )

    return append_ohlcv(df)


def sync_from_supabase(
    supabase: Client,
    tickers: List[str],
) -> int:
    init_db()

    total = 0
    for t in tickers:
        try:
            n = sync_ticker_from_supabase(supabase, t)
            total += n
        except Exception as e:
            print(f"[WARN] {t}: {e}")

    return total


if __name__ == "__main__":

    supabase = supabase_client()

    n = sync_from_supabase(
        supabase,
        get_all_tickers(),
    )

    print(f"Pulled {n} rows from Supabase into SQLite")
