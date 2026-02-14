import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv
from supabase import create_client, Client

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "universe.db"
# DB_PATH.parent.mkdir(exist_ok=True)

@contextmanager
def sqlite_connection(db_path: Path | None = None):
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def load_env():
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)


def supabase_client() -> Client:
    load_env()
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_API_KEY"]
    return create_client(url, key)
