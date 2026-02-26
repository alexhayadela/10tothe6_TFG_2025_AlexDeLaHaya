import pandas as pd
from sqlite3 import Connection

from db.base import sqlite_connection

def _get_last_date(conn: Connection) -> str | None:
    """Return the latest stored news date."""
    cur = conn.execute("SELECT MAX(date) FROM news")
    row = cur.fetchone()

    return row[0]
    

def load_news(start=None, end=None) -> pd.DataFrame:
    """Fetch ohlcv from a list of tickers."""
    query = "SELECT * FROM news "
    
    params = None

    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)

    with sqlite_connection() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["date"])
    return df



def load_entities(start=None, end=None) -> pd.DataFrame:
    """Fetch ohlcv from a list of tickers."""
    query = "SELECT * FROM news_entities "
    
    params = None

    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)

    with sqlite_connection() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["date"])
    return df
