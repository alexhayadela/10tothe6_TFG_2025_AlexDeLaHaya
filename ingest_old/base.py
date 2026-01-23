from pathlib import Path
import sqlite3
from contextlib import contextmanager


# Connection
@contextmanager
def sqlite_connection(db_path: Path):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
