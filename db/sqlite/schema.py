from sqlite3 import Connection

from db.base import sqlite_connection


def init_db(conn: Connection) -> None:
    """Initialize all tables from database."""
    create_news_table(conn)
    create_entities_table(conn)
    create_ohlcv_table(conn)


def create_news_table(conn: Connection) -> None:
    """Initialize news table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            section TEXT,
            date DATE NOT NULL,
            title TEXT NOT NULL,
            body TEXT,
            url TEXT NOT NULL UNIQUE,
            category TEXT,
            sentiment TEXT,
            relevance REAL
        );""")
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_date ON news(date);")
    

def create_entities_table(conn: Connection) -> None:
    """Initialize entities table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_entities (
            news_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            PRIMARY KEY (news_id, ticker),
            FOREIGN KEY (news_id) REFERENCES news(id) ON DELETE CASCADE
        );""")
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_entities_ticker ON news_entities(ticker);")


def create_ohlcv_table(conn: Connection) -> None:
    """Initialize ohlcv table."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        );""")
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv(date);")


if __name__ == "__main__":
    with sqlite_connection() as conn:
        init_db(conn)

