import pandas as pd
from pathlib import Path
from supabase import Client
from sqlite3 import Connection

from utils import load_env
from db.base import supabase_client, sqlite_connection
from db.utils_ohlcv import get_all_tickers
from db.sqlite.ingest_news import ingest_news 
from db.sqlite.queries_news import _get_last_date as get_last_date_news
from db.sqlite.ingest_ohlcv import ingest_ohlcv
from db.sqlite.queries_ohlcv import _get_last_date as get_last_date_ohlcv
from db.supabase.queries_news import _fetch_news_since
from db.supabase.queries_ohlcv import fetch_ohlcv_since


def news_migration(supabase: Client, conn: Connection) -> None:
    """Migrate news from supabase -> sqlite."""
    news_items = _fetch_news_since(supabase, get_last_date_news())
    ingest_news(conn, news_items)


def ohlcv_migration(supabase: Client, conn: Connection) -> None:
    """Migrate ohlcv from supabase -> sqlite."""
    dfs = []

    tickers = get_all_tickers()
    for t in tickers:
        last_date = get_last_date_ohlcv
        df = fetch_ohlcv_since(supabase, t, last_date)
        if not df.empty:
             df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d") # Sqlite compatible
             dfs.append(df)
    
    if dfs: 
         final_df = pd.concat(dfs, ignore_index=True)
         ingest_ohlcv(conn, final_df)


if __name__ == "__main__":
    load_env()

    supabase = supabase_client()
    with sqlite_connection() as conn:
        news_migration(supabase, conn)
        ohlcv_migration(supabase, conn)