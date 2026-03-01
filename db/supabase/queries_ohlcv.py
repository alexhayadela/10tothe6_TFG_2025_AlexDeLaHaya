import datetime
import pandas as pd
from supabase import Client

from db.base import supabase_client
from db.utils_ohlcv import get_ibex_tickers

def _get_last_date(supabase: Client, ticker: str) -> str | None:
    """Return the latest stored ohlcv date."""
    res = (
        supabase
        .table("ohlcv")
        .select("date")
        .eq("ticker", ticker)
        .order("date", desc=True)
        .limit(1)
        .execute())

    if res.data:
        return res.data[0]["date"]
    return None


def fetch_ohlcv(tickers: list[str], count:int) -> pd.DataFrame:
    """Fetch {count} ohlcv rows from a list of tickers."""  
    supabase = supabase_client()
    res = supabase.rpc("get_last_n_per_ticker", {"n_rows":count, "tickers": tickers}).execute()
    
    if not res.data:
        return pd.DataFrame()

    return pd.DataFrame(res.data)                 


def fetch_ohlcv_since(supabase: Client, ticker: str, start_date: str | None = None) -> pd.DataFrame:
    """Fetch ohlcv from a list of tickers."""
    query = (
        supabase
        .table("ohlcv")
        .select("ticker,date,open,high,low,close,volume")
        .eq("ticker", ticker)
        .order("date", desc=False)
    )

    if start_date:
        query = query.gt("date", start_date)

    res = query.execute()

    if not res.data:
        return pd.DataFrame()
    
    return  pd.DataFrame(res.data)


def top_k_predictions(k:int, date: datetime.date) -> pd.DataFrame:
    supabase = supabase_client()
    tickers = get_ibex_tickers()
    
    query = (
        supabase
        .table("predictions")
        .select("ticker,date,pred,proba")
        .eq("date", date)
        .in_("ticker", tickers)
        .order("proba", desc=True)
        .limit(k)
    )
    res = query.execute()

    if not res.data:
        return pd.DataFrame()
    
    return  pd.DataFrame(res.data)