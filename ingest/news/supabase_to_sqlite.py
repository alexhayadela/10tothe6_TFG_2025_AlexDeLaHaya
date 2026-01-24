from pathlib import Path
from ingest.news.ingest_sqlite import get_last_date
from ingest.base import supabase_client, sqlite_connection


DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "news.db"


def fetch_supabase_news(supabase, since_date: str = None) -> list[dict]:
    """
    Fetch news from Supabase with nested entities.
    since_date should be a string "YYYY-MM-DD" to filter newer news.
    """
    query = supabase.table("news").select("""
        id,
        section,
        date,
        title,
        body,
        url,
        category,
        sentiment_gpt,
        sentiment,
        relevance,
        news_entities(ticker)
    """)

    if since_date:
        query = query.gt("date", since_date)

    res = query.order("date").execute()
    return res.data or []


def ingest_news(supabase, since_date: str = None):
    news_list = fetch_supabase_news(supabase, since_date)

    with sqlite_connection(DB_PATH) as conn:
        for n in news_list:
            # --- Upsert news ---
            cur = conn.execute("""
                INSERT INTO news (
                    section, date, title, body, url,
                    category, sentiment_gpt, sentiment, relevance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    section=excluded.section,
                    date=excluded.date,
                    title=excluded.title,
                    body=excluded.body,
                    category=excluded.category,
                    sentiment_gpt=excluded.sentiment_gpt,
                    sentiment=excluded.sentiment,
                    relevance=excluded.relevance
            """, (
                n["section"],
                n["date"],
                n["title"],
                n["body"],
                n["url"],
                n["category"],
                n["sentiment_gpt"],
                n["sentiment"],
                n["relevance"]
            ))

            # Get SQLite ID for entities
            news_id = cur.lastrowid
            if news_id == 0:  # row already existed
                news_id = conn.execute(
                    "SELECT id FROM news WHERE url = ?",
                    (n["url"],)
                ).fetchone()[0]

            # --- Insert news_entities ---
            for e in n.get("news_entities", []):
                conn.execute("""
                    INSERT OR IGNORE INTO news_entities (news_id, ticker)
                    VALUES (?, ?)
                """, (news_id, e["ticker"]))

        conn.commit()
    return len(news_list)


if __name__ == "__main__":
    
    last_date = get_last_date()
    supabase = supabase_client()

    inserted = ingest_news(supabase, last_date)
    print(f"Supabase → SQLite synced {inserted} news since {last_date}")
