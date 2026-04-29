import pandas as pd
import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch


def _mock_supabase():
    client = MagicMock()
    return client


# -- queries_ohlcv (supabase) -------------------------------------------------

def test_supabase_fetch_ohlcv_calls_rpc():
    from db.supabase.queries_ohlcv import fetch_ohlcv
    client = _mock_supabase()
    client.rpc.return_value.execute.return_value.data = [
        {"ticker": "BBVA.MC", "date": "2024-01-15", "open": 7.1,
         "high": 7.2, "low": 7.0, "close": 7.15, "volume": 500000.0}
    ]
    with patch("db.supabase.queries_ohlcv.supabase_client", return_value=client):
        result = fetch_ohlcv(["BBVA.MC"], count=50)
    client.rpc.assert_called_once()
    assert isinstance(result, pd.DataFrame)


def test_supabase_fetch_ohlcv_returns_dataframe():
    from db.supabase.queries_ohlcv import fetch_ohlcv
    client = _mock_supabase()
    client.rpc.return_value.execute.return_value.data = []
    with patch("db.supabase.queries_ohlcv.supabase_client", return_value=client):
        result = fetch_ohlcv(["BBVA.MC"], count=10)
    assert isinstance(result, pd.DataFrame)


# -- ingest_ohlcv (supabase) --------------------------------------------------

def test_supabase_ingest_ohlcv_calls_upsert():
    from db.supabase.ingest_ohlcv import ingest_ohlcv
    client = _mock_supabase()
    df = pd.DataFrame({
        "ticker": ["BBVA.MC"], "date": ["2024-01-15"],
        "open": [7.1], "high": [7.2], "low": [7.0], "close": [7.15], "volume": [500000.0]
    })
    ingest_ohlcv(client, df)
    client.table.assert_called()
    client.table.return_value.upsert.assert_called()


# -- upload_preds (supabase) --------------------------------------------------

def test_upload_preds_calls_upsert():
    from db.supabase.upload_preds import upload_preds
    client = _mock_supabase()
    df = pd.DataFrame({
        "ticker": ["BBVA.MC"], "date": ["2024-01-15"],
        "pred": [1], "proba": [0.72], "model": ["rf_h1_sliding_discrete"]
    })
    upload_preds(client, df)
    client.table.assert_called_with("predictions")
    client.table.return_value.upsert.assert_called()


def test_upload_preds_empty_df_does_not_call_upsert():
    from db.supabase.upload_preds import upload_preds
    client = _mock_supabase()
    upload_preds(client, pd.DataFrame())
    client.table.assert_not_called()


# -- queries_news (supabase) --------------------------------------------------

def test_supabase_top_k_news_calls_select():
    from db.supabase.queries_news import top_k_news
    client = _mock_supabase()
    chain = (client.table.return_value.select.return_value
             .eq.return_value.order.return_value.limit.return_value.execute.return_value)
    chain.data = [{"title": "T", "body": "B", "url": "http://x.com"}]
    with patch("db.supabase.queries_news.supabase_client", return_value=client):
        result = top_k_news(k=5, date="2024-01-15")
    client.table.assert_called()
    assert isinstance(result, list)
