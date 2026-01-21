import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
import yfinance as yf

from ingest.base import sqlite_connection
from ingest.utils import get_ibex_tickers

# Configuration
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ibex35.db"
DB_PATH.parent.mkdir(exist_ok=True)


# Schema
def init_db():
    with sqlite_connection(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv(date);"
        )


# Ingestion
def get_last_date(ticker: str):
    with sqlite_connection(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,)
        )
        row = cur.fetchone()
        return row[0]


def download_ohlcv(ticker: str, start=None):
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, auto_adjust=False)
    if df.empty:
        return df

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["ticker"] = ticker
    return df[["ticker", "date", "open", "high", "low", "close", "volume"]]


def append_ohlcv(df: pd.DataFrame):
    if df.empty:
        return 0
    with sqlite_connection(DB_PATH) as conn:
        df.to_sql("ohlcv", conn, if_exists="append", index=False)
    return len(df)


def update_ticker(ticker: str):
    last_date = get_last_date(ticker)
    start = None
    if last_date:
        start = pd.to_datetime(last_date) + pd.Timedelta(days=1)

    df = download_ohlcv(ticker, start=start)
    return append_ohlcv(df)


def update_tickers(tickers: List[str]):
    init_db()
    inserted = 0
    for t in tickers:
        try:
            inserted += update_ticker(t)
        except Exception:
            continue
    return inserted


# Queries
def load_ohlcv(
    tickers: List[str], start=None, end=None
) -> pd.DataFrame:
    query = "SELECT * FROM ohlcv WHERE ticker IN ({})".format(
        ",".join("?" * len(tickers))
    )
    params = tickers

    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)

    with sqlite_connection(DB_PATH) as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["date"])
    return df


if __name__ == "__main__":
   

    n = update_tickers(get_ibex_tickers())
    print(f"Inserted {n} new OHLCV rows")
    