import pandas as pd
import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch


def _mock_conn():
    conn = MagicMock()
    conn.__enter__ = lambda s: s
    conn.__exit__ = MagicMock(return_value=False)
    return conn


# -- queries_ohlcv (sqlite) ---------------------------------------------------

def test_sqlite_get_last_date_returns_value():
    from db.sqlite.queries_ohlcv import _get_last_date
    conn = _mock_conn()
    conn.execute.return_value.fetchone.return_value = ("2024-01-15",)
    result = _get_last_date(conn, "BBVA.MC")
    assert result == "2024-01-15"


def test_sqlite_get_last_date_returns_none():
    from db.sqlite.queries_ohlcv import _get_last_date
    conn = _mock_conn()
    conn.execute.return_value.fetchone.return_value = (None,)
    result = _get_last_date(conn, "BBVA.MC")
    assert result is None


# -- queries_news (sqlite) ----------------------------------------------------

def test_load_news_no_filter_query():
    """No filters -> no WHERE clause in query."""
    with patch("db.sqlite.queries_news.sqlite_connection") as mock_ctx:
        mock_conn = MagicMock()
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        with patch("db.sqlite.queries_news.pd.read_sql", return_value=pd.DataFrame()) as mock_sql:
            from db.sqlite.queries_news import load_news
            load_news()
            query_used = mock_sql.call_args[0][0]
            assert "WHERE" not in query_used.upper()


def test_load_news_start_filter_query():
    """start provided -> WHERE date >= in query."""
    with patch("db.sqlite.queries_news.sqlite_connection") as mock_ctx:
        mock_conn = MagicMock()
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        with patch("db.sqlite.queries_news.pd.read_sql", return_value=pd.DataFrame()) as mock_sql:
            from db.sqlite.queries_news import load_news
            load_news(start="2024-01-01")
            query_used = mock_sql.call_args[0][0]
            assert "date >=" in query_used.lower()


def test_load_news_both_filters_query():
    """Both start and end -> two conditions in query."""
    with patch("db.sqlite.queries_news.sqlite_connection") as mock_ctx:
        mock_conn = MagicMock()
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        with patch("db.sqlite.queries_news.pd.read_sql", return_value=pd.DataFrame()) as mock_sql:
            from db.sqlite.queries_news import load_news
            load_news(start="2024-01-01", end="2024-01-31")
            query_used = mock_sql.call_args[0][0]
            assert "date >=" in query_used.lower()
            assert "date <=" in query_used.lower()


def test_load_entities_no_filter_runs():
    with patch("db.sqlite.queries_news.sqlite_connection") as mock_ctx:
        mock_conn = MagicMock()
        mock_ctx.return_value.__enter__ = lambda s: mock_conn
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        with patch("db.sqlite.queries_news.pd.read_sql", return_value=pd.DataFrame()):
            from db.sqlite.queries_news import load_entities
            load_entities()  # must not raise


# -- db/base sqlite_connection ------------------------------------------------

def test_sqlite_connection_context_manager():
    with patch("db.base.sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        from db.base import sqlite_connection
        with sqlite_connection():
            pass
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()


def test_sqlite_connection_closes_on_exception():
    with patch("db.base.sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        from db.base import sqlite_connection
        with pytest.raises(ValueError):
            with sqlite_connection():
                raise ValueError("test error")
        mock_conn.close.assert_called_once()
