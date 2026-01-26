import feedparser
import re 
import datetime


def fetch_rss() -> list[dict]:
    feeds = {
        "mercados": "https://e01-expansion.uecdn.es/rss/mercados.xml",
        "ahorro": "https://e01-expansion.uecdn.es/rss/ahorro.xml",
        "economia": "https://e01-expansion.uecdn.es/rss/empresas.xml",
        "empresas": "https://e01-expansion.uecdn.es/rss/empresas.xml"
    }

    items = []
    rss_format = "%a, %d %b %Y %H:%M:%S %z"

    for section, url in feeds.items():
        feed = feedparser.parse(url)

        for entry in feed.entries:
            summary = re.sub("<.*?>", "", entry.get("summary", ""))
            summary = summary.replace("&nbsp;Leer", " ").strip()
            date = datetime.datetime.strptime(
                entry.get("published"),
                rss_format
            ).strftime("%Y-%m-%d")
    
            items.append({
                "source": "expansion",
                "section": section,
                "title": entry.get("title"),
                "summary": summary,
                "link": entry.get("link"),
                "date": date,
                "tags": [t.get("term") for t in entry.get("tags", [])]
            })

    # dedupe by link
    unique = {item["link"]: item for item in items}
    return list(unique.values())


def last_news() -> list[dict]:
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    items = fetch_rss()
    filtered = [
        item for item in items
        if item["date"] == yesterday
    ]
    return filtered


def top_news(items: list[dict],k) -> list[dict]:
    top_news = sorted(
    items,
    key=lambda x: x["relevance_score"],
    reverse=True
)[:k]

    return top_news


def newsletter_ready(news_list: list[dict]) -> list[dict]:
    return [
        {
            "title": n["title"],
            "summary": n["summary"],
            "link": n["link"],
        }
        for n in news_list
    ]


def html_news(items: list[dict]) -> str:
    html = ""
    for i, item in enumerate(items, 1):
        html += f"""<div class="news-item">
            <h2>{i}. {item['title']}</h2>
            <p>{item['summary']} 
                <a href="{item['link']}">See more</a>
            </p>
        </div>
        """
    return html


# Test
if __name__ == "__main__" :

    news = last_news()
    print(news)
    
    news_html = html_news(news)
    print(news_html)