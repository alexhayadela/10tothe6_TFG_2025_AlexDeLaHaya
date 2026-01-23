import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv
from supabase import create_client, Client


@contextmanager
def sqlite_connection(db_path: Path):
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



