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
    """Fetch ohlcv from a list of tickers."""
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

