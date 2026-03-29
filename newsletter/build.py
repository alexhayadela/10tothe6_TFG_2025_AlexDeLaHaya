import datetime
import pandas as pd

from config import load_env
from db.supabase.queries_news import top_k_news
from db.supabase.queries_ohlcv import top_k_predictions
from db.utils_ohlcv import ticker_to_name


def add_header(date: datetime.date):
    html = """<!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>10**6 Boletín</title>

    <style>
    body {
        margin:0;
        padding:0;
        background-color:#ffffff;
        font-family: Arial, Helvetica, sans-serif;
    }

    .container {
        max-width:650px;
        margin:30px auto;
        background-color:#f0f0f0;
        border-radius:12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding:30px;
    }

    .header {
        text-align:center;
    }

    h1 {
        text-align:center;
        color:darkblue;
        font-size:28px;
        margin:30px 0;
    }

    h2 {
        color: darkblue;
        font-size:18px;
        margin:0;
    }

    p {
        color: black;
        font-size: 14px;
        line-height:1.6;
    }

    footer {
        text-align:center;
        font-size:12px;
        margin-top:10px;
    }

    .news-item {
        margin-bottom:20px;
        text-align: justify;
    }

    .predictions-table {
        width:100%;
        margin-bottom:30px;
    }

    .predictions-table td {
        padding:0 9px;
        vertical-align:top;
    }

    .signature{
        display:block;
        width:120px;       
        height:auto;
        margin:30px auto 10px auto;  
    }
    </style>
    </head>

    <body>

    <div class="container">

    <div class="header">
        <img src="cid:freakbob" alt=freakbob
            width="479" height="242"
            style="display:block; margin:0 auto; border-radius:6px;">
    </div>
"""

    html += f"""<h1>Informe diario — {date.day}/{date.month}</h1>"""

    return html


def add_footer(date: datetime.date) -> str:
    html = f"""
    <img class="signature" src="cid:67" alt="67"/>
    <footer>
    <p>Alex De La Haya © {date.year} | 10**6, Boletín</p>
    </footer>"""
    return html


def add_closing() -> str:
    html = """</div>
    </body>
    </html>"""
    return html


def add_news(items: list[dict]) -> str:
    html = ""
    for i, item in enumerate(items, 1):
        html += f"""<div class="news-item">
            <h2>{i}. {item['title']}</h2>
            <p>{item['body']} 
                <a href="{item['url']}">See more</a>
            </p>
        </div>
        """
    return html


def format_predictions(df_pred: pd.DataFrame) -> pd.DataFrame:
    df_pred["name"] = df_pred["ticker"].map(ticker_to_name())
    df_pred["action"] = df_pred["pred"].map({True: "Buy", False: "Sell"})
    df_pred["proba.2f"] = df_pred["proba"].round(2)
    return df_pred


def add_predictions(items: pd.DataFrame) -> str:
    html = """<table class="predictions-table" cellpadding="0" cellspacing="0">
    <tr>"""
    for _, item in items.iterrows():
        html += f"""
        <td width="33%" align="center">
    <table width="100%" cellpadding="0" cellspacing="0"
           style="background:#ffffff;
                  border-radius:10px;
                  box-shadow:0 4px 12px rgba(0,0,0,0.08);
                  border:1px solid #e2e2e2;">

        <tr>
            <td align="center" style="padding:20px 10px;">
            <h2 style="line-height:1;">{item["name"]}</h2>
        </td>
        </tr>
        <tr>
            <td align="center"
                bgcolor="#5fcf92"
                style="padding:12px 10px;
                       font-family:'Courier New', monospace;
                       font-weight:bold;">
            P({item["action"]}|Xₜ)= {item["proba.2f"]}
        </td>
        </tr>
        </table>
        </td>"""
    html += """</tr>
    </table>"""

    return html


def build_newsletter() -> str:
    today = datetime.date.today()
    weekday = today.weekday()
    rel_date = today - datetime.timedelta(days=1)

    header = add_header(today)
    footer = add_footer(today)
    closing = add_closing()

    news = top_k_news(k=10, date=str(rel_date))
    news_html = add_news(news)

    # market closed on weekends
    if weekday not in {5, 6}:
        # handle fri <- mon
        rel_date = rel_date - datetime.timedelta(days=2) if weekday == 0 else rel_date

        preds = top_k_predictions(k=3, date=str(rel_date))
        preds = format_predictions(preds)
        preds_html = add_predictions(preds)
        html = header + preds_html + news_html + footer + closing

    else:
        html = header + news_html + footer + closing

    return html


if __name__ == "__main__":
    load_env()

    newsletter = build_newsletter()
    print(newsletter)
