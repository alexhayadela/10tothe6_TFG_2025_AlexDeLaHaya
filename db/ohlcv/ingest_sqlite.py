import pandas as pd
from typing import List


from db.base import sqlite_connection
from db.ohlcv.utils import get_ibex_tickers, download_ticker


def init_db():
    with sqlite_connection() as conn:
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


def get_last_date(ticker: str):
    with sqlite_connection() as conn:
        cur = conn.execute(
            "SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,)
        )
        row = cur.fetchone()
        return row[0]


def append_ohlcv(df: pd.DataFrame):
    if df.empty:
        return 0
    with sqlite_connection() as conn:
        df.to_sql("ohlcv", conn, if_exists="append", index=False)
    return len(df)


def update_ticker(ticker: str):
    last_date = get_last_date(ticker)
    start = None
    if last_date:
        start = pd.to_datetime(last_date) + pd.Timedelta(days=1)

    df = download_ticker(ticker, start=start)
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


if __name__ == "__main__":
   
    n = update_tickers(get_ibex_tickers())
    print(f"Inserted {n} new OHLCV rows")
    