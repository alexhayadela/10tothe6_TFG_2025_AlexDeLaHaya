import pandas as pd
from sqlite3 import Connection


from db.base import sqlite_connection
from db.utils_ohlcv import get_all_tickers, download_ticker
from db.sqlite.queries_ohlcv import _get_last_date


def ingest_ohlcv(conn: Connection, df: pd.DataFrame) -> None:
    """Insert OHLCV rows into the database."""
    df.to_sql("ohlcv", conn, if_exists="append", index=False)


def update_ticker(conn: Connection, ticker: str) -> pd.DataFrame:
    """Incrementally download missing OHLCV bars for a single ticker.

    Checks the latest stored date in SQLite; if found, fetches only bars
    from the next day onwards to avoid re-downloading existing data.
    If no data exists yet, fetches the full history. Returns the new rows
    as a DataFrame (empty if already up to date).
    """
    last_date = _get_last_date(conn, ticker)
    start = None
    if last_date:
        start = pd.to_datetime(last_date) + pd.Timedelta(days=1)

    df = download_ticker(ticker, start=start)
    return df


def update_tickers(conn: Connection, tickers: list[str]) -> pd.DataFrame:
    """Incrementally update a list of tickers and return the combined new rows.

    Calls update_ticker for each ticker, skipping any that raise an exception
    (e.g. delisted symbols). Concatenates all non-empty results into a single
    DataFrame ready to be passed to ingest_ohlcv.
    """
    dfs = []
    for t in tickers:
        try:
            df = update_ticker(conn, t)
            if not df.empty:
                dfs.append(df)
        except Exception:  # A ticker may be decomissioned.
            continue

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def ingest_predictions(conn: Connection, df: pd.DataFrame) -> None:
    """Insert prediction rows into the database."""
    df.to_sql("predictions", conn, if_exists="append", index=False)


if __name__ == "__main__":
    tickers = get_all_tickers()
    with sqlite_connection() as conn:
        df = update_tickers(conn, tickers)
        ingest_ohlcv(conn, df)
