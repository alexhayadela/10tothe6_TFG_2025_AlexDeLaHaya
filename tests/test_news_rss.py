import datetime
import pytest
import unittest.mock as mock
from news.news_rss import last_news, top_news


def _make_items(dates_and_relevances):
    return [{"title": f"T{i}", "body": "B", "url": f"http://{i}.com",
             "date": d, "relevance": r, "source": "exp", "section": "s", "tags": []}
            for i, (d, r) in enumerate(dates_and_relevances)]


def test_last_news_keeps_yesterday():
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    older     = (datetime.date.today() - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
    items = _make_items([(yesterday, 0.5), (older, 0.8), (yesterday, 0.3)])
    with mock.patch("news.news_rss.fetch_rss", return_value=items):
        result = last_news()
    assert all(item["date"] == yesterday for item in result)
    assert len(result) == 2


def test_last_news_empty_when_no_yesterday_items():
    older = (datetime.date.today() - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
    items = _make_items([(older, 0.5)])
    with mock.patch("news.news_rss.fetch_rss", return_value=items):
        result = last_news()
    assert result == []


def test_top_news_returns_k_items():
    items = _make_items([("2024-01-01", r) for r in [0.1, 0.5, 0.9, 0.3, 0.7]])
    result = top_news(items, k=3)
    assert len(result) == 3


def test_top_news_sorted_descending():
    items = _make_items([("2024-01-01", r) for r in [0.1, 0.9, 0.5]])
    result = top_news(items, k=3)
    relevances = [item["relevance"] for item in result]
    assert relevances == sorted(relevances, reverse=True)
