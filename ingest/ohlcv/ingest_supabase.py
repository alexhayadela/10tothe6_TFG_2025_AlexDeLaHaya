import pandas as pd
from typing import List
from ingest.base import supabase_client
from ingest.utils import get_ibex_tickers, download_tickers
import datetime


def get_last_date(ticker: str):
    res = (
        supabase_client()
        .table("ohlcv")
        .select("date")
        .eq("ticker", ticker)
        .order("date", desc=True)
        .limit(1)
        .execute()
    )

    if res.data:
        return res.data[0]["date"]
    return None


def append_ohlcv(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    records = df.to_dict(orient="records")

    res = (
        supabase_client()
        .table("ohlcv")
        .upsert(
            records,
            on_conflict="ticker,date"
        )
        .execute()
    )

    return len(records)


def update_ticker(ticker: str):
    last_date = get_last_date(ticker)
    last_date = datetime.date.today() - datetime.timedelta(days=3)
    start = None
    if last_date:
        start = pd.to_datetime(last_date) + pd.Timedelta(days=1)

    df = download_tickers(ticker, start=start)
    return append_ohlcv(df)


def update_tickers(tickers: List[str]):
    inserted = 0

    for t in tickers:
        try:
            inserted += update_ticker(t)
        except Exception as e:
            print(f"[WARN] {t}: {e}")
            continue

    return inserted

if __name__ == "__main__":
    n = update_tickers(get_ibex_tickers())
    print(f"Inserted {n} new OHLCV rows into Supabase")

