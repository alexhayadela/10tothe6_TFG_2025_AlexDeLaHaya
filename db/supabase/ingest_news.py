from supabase import Client

from utils import load_env
from db.base import supabase_client
from news.classification import classify_news
from news.news_rss import last_news


def ingest_news(supabase: Client, news_list: list[dict]) -> None:
    """
    Store classified news items and related entities in the database.
    """
    for n in news_list:
        res = supabase.table("news").upsert({
            "section": n.get("section"),
            "date": n.get("date"),
            "title": n.get("title"),
            "body": n.get("body"),
            "url": n.get("url"),
            "category": n.get("category"),
            "sentiment": n.get("sentiment"),
            "relevance": n.get("relevance")
        }, on_conflict="url").execute()

        # Fetch the assigned ID (needed for entities)
        news_id = res.data[0]["id"] if res.data else None
        if not news_id:
            # fallback: fetch ID by URL
            news_id = supabase.table("news").select("id").eq("url", n["url"]).execute().data[0]["id"]

        for company in n.get("companies", []):
            supabase.table("news_entities").upsert({
                "news_id": news_id,
                "ticker": company
            }, on_conflict=["news_id,ticker"]).execute()


if __name__ == "__main__":
    load_env()
    
    news = last_news()
    classified_news = classify_news(news)

    supabase = supabase_client()
    ingest_news(supabase, classified_news)
