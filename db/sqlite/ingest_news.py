from sqlite3 import Connection

from utils import load_env
from db.base import sqlite_connection
from news.classification import classify_news
from news.news_rss import last_news


def ingest_news(conn: Connection, news_items: list[dict]) -> None:
    """
    Store classified news items and related entities in the database.
    """
    for n in news_items:

        cur = conn.execute("""
            INSERT INTO news (
                section,
                date,
                title,
                body,
                url,
                category,
                sentiment,
                relevance
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                section=excluded.section,
                date=excluded.date,
                title=excluded.title,
                body=excluded.body,
                category=excluded.category,
                sentiment=excluded.sentiment,
                relevance=excluded.relevance
        """,
        (
            n.get("section"),
            n.get("date"),
            n.get("title"),
            n.get("body"),
            n.get("url"),
            n.get("category"),
            n.get("sentiment"),
            n.get("relevance")))

        news_id = conn.execute(
                "SELECT id FROM news WHERE url = ?",
                (n["url"],)
            ).fetchone()[0]

        # Support both formats safely (ingest from supabase or local)
        entities = n.get("news_entities") or n.get("companies") or []

        for e in entities:
            ticker = e["ticker"] if isinstance(e, dict) else e

            conn.execute("""
                INSERT OR IGNORE INTO news_entities (news_id, ticker)
                VALUES (?, ?)
            """, (news_id, ticker))

    conn.commit()

    
if __name__ == "__main__":
    load_env()
    
    news = last_news()
    classified_news = classify_news(news)
    
    with sqlite_connection() as conn:
        ingest_news(conn, classified_news)