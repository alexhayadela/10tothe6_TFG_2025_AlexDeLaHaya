import re 
import datetime
import feedparser


def fetch_rss() -> list[dict]:
    """Fetch news items from rss."""
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
            body = re.sub("<.*?>", "", entry.get("summary", ""))
            body = body.replace("&nbsp;Leer", " ").strip()
            date = datetime.datetime.strptime(
                entry.get("published"),
                rss_format
            ).strftime("%Y-%m-%d")
    
            items.append({
                "source": "expansion",
                "section": section,
                "title": entry.get("title"),
                "body": body,
                "url": entry.get("link"),
                "date": date,
                "tags": [t.get("term") for t in entry.get("tags", [])]
            })

    # dedupe by link
    unique = {item["url"]: item for item in items}
    return list(unique.values())


def last_news() -> list[dict]:
    """Returns news from now <-> 24h earlier."""
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    items = fetch_rss()
    filtered = [
        item for item in items
        if item["date"] == yesterday
    ]
    return filtered


def top_news(items: list[dict],k) -> list[dict]:
    """Sorts news items by relevance."""
    top_news = sorted(
    items,
    key=lambda x: x["relevance"],
    reverse=True
)[:k]

    return top_news


if __name__ == "__main__" :

    news = last_news()
    print(news)
    