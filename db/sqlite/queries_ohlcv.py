import pandas as pd
from sqlite3 import Connection

from db.base import sqlite_connection


def _get_last_date(conn: Connection, ticker: str) -> str | None:
    """Returns last date for a ticker."""
    cur = conn.execute(
        "SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    return row[0]


def fetch_ohlcv(tickers: list[str], start=None, end=None) -> pd.DataFrame:
    """Load OHLCV rows from SQLite for the given tickers and optional date range.

    Builds a parameterised query at runtime: always filters by ticker list,
    appends AND date >= start and/or AND date <= end when provided.
    Returns a DataFrame with columns: ticker, date (parsed), open, high, low,
    close, volume. Returns all history when start/end are omitted — useful for
    training; pass start/end to get a recent window for inference.
    """
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

    with sqlite_connection() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["date"])
    return df

def _get_last_date_predictions(conn: Connection, ticker: str) -> str | None:
    """Returns last date for a ticker prediction."""
    cur = conn.execute(
        "SELECT MAX(date) FROM predictions WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    return row[0]