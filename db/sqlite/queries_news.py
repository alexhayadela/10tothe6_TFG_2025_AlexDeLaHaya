from sqlite3 import Connection


def _get_last_date(conn: Connection) -> str | None:
    """Return the latest stored news date."""
    cur = conn.execute("SELECT MAX(date) FROM news")
    row = cur.fetchone()

    return row[0]
    
