from typing import List
import pandas as pd 
from db.base import sqlite_connection
from pathlib import Path 

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ibex35.db"

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