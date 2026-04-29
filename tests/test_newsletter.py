import datetime
import pandas as pd
import pytest
import unittest.mock as mock
from newsletter.build import (
    add_header,
    add_footer,
    add_closing,
    add_news,
    add_predictions,
    format_predictions,
)


@pytest.fixture
def today():
    return datetime.date(2024, 6, 15)


@pytest.fixture
def news_items():
    return [
        {"title": "BBVA sube", "body": "El banco sube un 2%.", "url": "http://a.com"},
        {"title": "Iberdrola cae", "body": "Baja un 1%.", "url": "http://b.com"},
    ]


@pytest.fixture
def pred_df():
    df = pd.DataFrame({
        "ticker": ["BBVA.MC", "SAN.MC", "ITX.MC"],
        "pred":   [True, False, True],
        "proba":  [0.72, 0.38, 0.65],
        "date":   ["2024-06-14"] * 3,
    })
    return df


def test_add_header_contains_date(today):
    html = add_header(today)
    assert str(today.day) in html
    assert str(today.month) in html


def test_add_header_is_html(today):
    html = add_header(today)
    assert "<html" in html.lower()


def test_add_footer_contains_year(today):
    html = add_footer(today)
    assert str(today.year) in html


def test_add_footer_is_html(today):
    html = add_footer(today)
    assert "<" in html


def test_add_closing_contains_body_close():
    html = add_closing()
    assert "</body>" in html
    assert "</html>" in html


def test_add_news_contains_titles(news_items):
    html = add_news(news_items)
    assert "BBVA sube" in html
    assert "Iberdrola cae" in html


def test_add_news_contains_urls(news_items):
    html = add_news(news_items)
    assert "http://a.com" in html


def test_format_predictions_adds_action(pred_df):
    with mock.patch("newsletter.build.ticker_to_name",
                    return_value={"BBVA.MC": "BBVA", "SAN.MC": "Santander", "ITX.MC": "Inditex"}):
        result = format_predictions(pred_df)
    assert "action" in result.columns
    assert set(result["action"].unique()).issubset({"Buy", "Sell"})


def test_format_predictions_adds_proba_col(pred_df):
    with mock.patch("newsletter.build.ticker_to_name",
                    return_value={"BBVA.MC": "BBVA", "SAN.MC": "Santander", "ITX.MC": "Inditex"}):
        result = format_predictions(pred_df)
    assert "proba.2f" in result.columns


def test_add_predictions_contains_ticker_names(pred_df):
    with mock.patch("newsletter.build.ticker_to_name",
                    return_value={"BBVA.MC": "BBVA", "SAN.MC": "Santander", "ITX.MC": "Inditex"}):
        formatted = format_predictions(pred_df)
    html = add_predictions(formatted)
    assert "BBVA" in html or "Santander" in html or "Inditex" in html
