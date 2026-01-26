from news.classification import classify_news
from news.news_rss import last_news
from ingest.base import supabase_client


def ingest_news_supabase(news_list: list[dict], supabase):
    for n in news_list:
        # 1. Insert or update news
        res = supabase.table("news").upsert({
            "section": n.get("section"),
            "date": n.get("date"),
            "title": n.get("title"),
            "body": n.get("summary"),
            "url": n.get("link"),
            "category": n.get("category"),
            "sentiment_gpt": n.get("sentiment"),
            "sentiment": None,
            "relevance": n.get("relevance_score")
        }, on_conflict="url").execute()

        # 2. Fetch the assigned ID (needed for entities)
        news_id = res.data[0]["id"] if res.data else None
        if not news_id:
            # fallback: fetch ID by URL
            news_id = supabase.table("news").select("id").eq("url", n["link"]).execute().data[0]["id"]

        # 3. Insert entities
        for company in n.get("companies", []):
            supabase.table("news_entities").upsert({
                "news_id": news_id,
                "ticker": company
            }, on_conflict=["news_id,ticker"]).execute()


if __name__ == "__main__":
   
    news = last_news()
    
    classified_news = classify_news(news)
    supabase = supabase_client()
    ingest_news_supabase(classified_news, supabase)
