import pandas as pd
from supabase import Client

from db.base import supabase_client
from db.utils_ohlcv import get_ibex_tickers


def _get_last_date(supabase: Client, ticker: str) -> str | None:
    """Return the most recent date stored in the ohlcv table for a ticker.

    Used by update_ticker to determine the incremental start point before
    downloading new bars from Yahoo Finance. Returns None if no data exists yet.
    """
    res = (
        supabase.table("ohlcv")
        .select("date")
        .eq("ticker", ticker)
        .order("date", desc=True)
        .limit(1)
        .execute()
    )

    if res.data:
        return res.data[0]["date"]
    return None


def fetch_ohlcv(tickers: list[str], count: int) -> pd.DataFrame:
<<<<<<< HEAD
    """Fetch {count} ohlcv rows from a list of tickers."""
    supabase = supabase_client()
    res = supabase.rpc(
        "get_last_n_per_ticker", {"n_rows": count, "tickers": tickers}
    ).execute()
=======
    """Fetch the last `count` OHLCV bars per ticker via a Supabase RPC call.

    Calls the `get_last_n_per_ticker` SQL function which returns the most
    recent `count` rows for each ticker in the list — not `count` total rows.
    This is the primary data loader for inference: pass count=50 to get
    enough history to compute all rolling indicators needed by the ML model.
    Returns a flat DataFrame with all tickers concatenated.
    """
    supabase = supabase_client()
    res = supabase.rpc("get_last_n_per_ticker", {"n_rows": count, "tickers": tickers}).execute()
>>>>>>> development-cl

    if not res.data:
        return pd.DataFrame()

    return pd.DataFrame(res.data)


<<<<<<< HEAD
def fetch_ohlcv_since(
    supabase: Client, ticker: str, start_date: str | None = None
) -> pd.DataFrame:
    """Fetch ohlcv from a ticker."""  # Can be done all at once.
=======
def fetch_ohlcv_since(supabase: Client, ticker: str, start_date: str | None = None) -> pd.DataFrame:
    """Fetch all OHLCV rows for one ticker, optionally filtered by start date.

    Returns rows strictly after `start_date` (exclusive) when provided,
    otherwise returns the full history. Results are ordered oldest-first,
    which is required by feature engineering functions that rely on sequential
    rolling calculations.
    """
>>>>>>> development-cl
    query = (
        supabase.table("ohlcv")
        .select("ticker,date,open,high,low,close,volume")
        .eq("ticker", ticker)
        .order("date", desc=False)
    )

    if start_date:
        query = query.gt("date", start_date)

    res = query.execute()

    if not res.data:
        return pd.DataFrame()

    return pd.DataFrame(res.data)


def top_k_predictions(k: int, date: str) -> pd.DataFrame:
<<<<<<< HEAD
=======
    """Return the k most confident predictions for a given date.

    Filters to IBEX35 tickers only, orders by probability descending, and
    returns the top k rows. Used by the newsletter to display the best
    buy/sell signals of the day.
    """
>>>>>>> development-cl
    supabase = supabase_client()
    tickers = get_ibex_tickers()

    query = (
        supabase.table("predictions")
        .select("ticker,date,pred,proba")
        .eq("date", date)
        .in_("ticker", tickers)
        .order("proba", desc=True)
        .limit(k)
    )
    res = query.execute()

    if not res.data:
        return pd.DataFrame()

    return pd.DataFrame(res.data)


<<<<<<< HEAD
def fetch_predictions_since(
    supabase: Client, ticker: str, start_date: str | None = None
) -> pd.DataFrame:
=======
def fetch_predictions_since(supabase: Client, ticker: str, start_date: str | None = None) -> pd.DataFrame:
    """Fetch prediction history for one ticker, optionally from a start date.

    Returns rows strictly after `start_date` (exclusive) when provided.
    Useful for backtesting or plotting prediction accuracy over time.
    """
>>>>>>> development-cl
    query = (
        supabase.table("predictions")
        .select("ticker,date,pred,proba,model")
        .eq("ticker", ticker)
        .order("date", desc=False)
    )

    if start_date:
        query = query.gt("date", start_date)

    res = query.execute()

    if not res.data:
        return pd.DataFrame()
<<<<<<< HEAD
=======

    return pd.DataFrame(res.data)

>>>>>>> development-cl

    return pd.DataFrame(res.data)
