import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
import yfinance as yf

from ingest.base import sqlite_connection
from news.classification import classify_news
from news.news_rss import last_news


# Configuration
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "news.db"
DB_PATH.parent.mkdir(exist_ok=True)


# Schema
def init_db():
    with sqlite_connection(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section TEXT,
                date DATE NOT NULL,
                title TEXT NOT NULL,
                body TEXT,
                url TEXT UNIQUE,
                category TEXT,
                sentiment_gpt TEXT,
                sentiment REAL,
                relevance REAL
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS news_entities (
                news_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                PRIMARY KEY (news_id, ticker),
                FOREIGN KEY (news_id) REFERENCES news(id) ON DELETE CASCADE
            );
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_date ON news(date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_category ON news(category);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_news_entities_ticker ON news_entities(ticker);")


def ingest_news(news_list):
    with sqlite_connection(DB_PATH) as conn:
        for n in news_list:
            
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO news (
                    section,
                    date,
                    title,
                    body,
                    url,
                    category,
                    sentiment_gpt,
                    sentiment,
                    relevance
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    n.get("section"),
                    n.get("date"),
                    n.get("title"),
                    n.get("summary"),
                    n.get("link"),
                    n.get("category"),
                    n.get("sentiment"),
                    None,  # numeric sentiment (later)
                    n.get("relevance_score"),
                )
            )

            news_id = cur.lastrowid
            if news_id == 0:
                news_id = conn.execute(
                    "SELECT id FROM news WHERE url = ?",
                    (n["link"],)
                ).fetchone()[0]

            # 3. Insert entities
            for company in n.get("companies", []):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO news_entities (news_id, ticker)
                    VALUES (?, ?)
                    """,
                    (news_id, company)
                )

if __name__ == "__main__":

    db = init_db()

    news = last_news()
    classified_news = classify_news(news)

    ingest_news(classified_news)
    