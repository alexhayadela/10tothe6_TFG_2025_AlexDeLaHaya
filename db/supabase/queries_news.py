import datetime
from supabase import Client

from db.base import supabase_client


def top_k_news(k: int, date: datetime.date) -> list[dict]:
    """Returns top-k most relevant news of that day."""
    supabase = supabase_client()
    query = (supabase
             .table("news")
             .select("title,body,url")
             .eq("date",date)
             .order("relevance", desc=True)
             .limit(k)
    )
    res = query.execute()
    
    return res.data


def _fetch_news_since(supabase: Client, since_date: str = None) -> list[dict]:
    """Fetch news with nested entities."""
    query = supabase.table("news").select("""
        id,
        section,
        date,
        title,
        body,
        url,
        category,
        sentiment,
        relevance,
        news_entities(ticker)
    """)

    if since_date:
        query = query.gt("date", since_date)

    res = query.order("date").execute()
    return res.data or []


def get_recipients() -> list[str]:
    """Returns newsletter recipients."""
    supabase = supabase_client()
    query = (supabase
            .table("newsletter")
            .select("email")
    )
    res = query.execute()
    
    # return appropiate format
    return [item["email"] for item in (res.data or [])]



