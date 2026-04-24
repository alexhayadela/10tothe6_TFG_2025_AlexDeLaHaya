import pandas as pd
from sqlite3 import Connection

from db.base import sqlite_connection


def _get_last_date(conn: Connection) -> str | None:
    """Return the latest stored news date."""
    cur = conn.execute("SELECT MAX(date) FROM news")
    row = cur.fetchone()

    return row[0]


def load_news(start=None, end=None) -> pd.DataFrame:
    """Fetch news rows with optional date range filter."""
    query = "SELECT * FROM news"
    params = []

    if start:
        query += " WHERE date >= ?"
        params.append(start)
    if end:
        query += (" AND" if start else " WHERE") + " date <= ?"
        params.append(end)

    with sqlite_connection() as conn:
        df = pd.read_sql(query, conn, params=params or None, parse_dates=["date"])
    return df


def load_entities(start=None, end=None) -> pd.DataFrame:
    """Fetch news_entities rows joined with news date, with optional date range filter."""
    query = (
        "SELECT ne.* FROM news_entities ne "
        "JOIN news n ON ne.news_id = n.id"
    )
    params = []

    if start:
        query += " WHERE n.date >= ?"
        params.append(start)
    if end:
        query += (" AND" if start else " WHERE") + " n.date <= ?"
        params.append(end)

    with sqlite_connection() as conn:
        df = pd.read_sql(query, conn, params=params or None)
    return df
