import feedparser
import re 
import pandas as pd
import datetime

def fetch_rss():
    # RSS feeds from Expansión
    feeds = {"mercados": "https://e01-expansion.uecdn.es/rss/mercados.xml", 
            "ahorro": "https://e01-expansion.uecdn.es/rss/ahorro.xml", 
            "economia": "https://e01-expansion.uecdn.es/rss/empresas.xml"}

    # Read / Process / Store RSS feeds data
    feeds_data = []
    for section, url in feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:

            # Get data
            title = entry.get("title")
            pre_summary = entry.get("summary")
            date = entry.get("published")
            link = entry.get("link")
            pre_tags = entry.get("tags")
            
            # Transform data: Process summary & tags
            pre_summary = re.sub("<.*?>", "", pre_summary)
            summary = pre_summary.replace("&nbsp;Leer", " ").strip()
            tags = []
            tags = []
            if pre_tags:
                for tag in pre_tags:
                    tags.append(tag.get("term"))  

            # Append data
            feeds_data.append({"section": section, "title": title, "summary": summary, "date": date, "link": link, "tags": tags })
            # Debug
            # print("Section: ", section)
            # print("Title: ", title)
            # print("Summary: ", summary)
            # print("Date: ", date)
            # # print("Link: ", link)
            # print("Tags: ", tags)

    df_feeds = pd.DataFrame(feeds_data)
    # Format date
    # df_feeds["date"]  = pd.to_datetime(df_feeds["date"]).dt.date (it failed due to hour change)
    rss_format = "%a, %d %b %Y %H:%M:%S %z"
    df_feeds["date"] = df_feeds["date"].apply(lambda x: datetime.datetime.strptime(x,rss_format).date())

    # Remove duplicates (rss feeds may overlap)
    df_feeds.drop_duplicates(subset=["title", "summary"])

    return df_feeds

def relevant_feeds():
    # Get most relevant feeds for todays newsletter
    df_feeds = fetch_rss()

    # Filter date to yesterday only (newsletter 9.00 am)
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    #df_feeds = df_feeds[df_feeds["date"].isin([today,yesterday])]
    df_feeds = df_feeds[df_feeds["date"] == yesterday]

    # Drop columns we don't want to show
    df_feeds = df_feeds.drop(columns=["date", "tags", "section"])
    #print(df_feeds.columns)

    # keep top 10 only (UPDATE LOGIC!!!)
    df_feeds = df_feeds.head(10)

    return df_feeds

def feeds_to_html(df_feeds):
    # Format feeds into html
    html = """"""
    for i,row in enumerate(df_feeds.itertuples(), start=1):
    
        html += f"""<div class="news-item">
            <h2>{i}. {row.title}</h2>
            <p>{row.summary} 
                <a href="{row.link}">See more</a>
            </p>
        </div>
    """

    return html

# Test
if __name__ == "__main__" :

    df_feeds = relevant_feeds()
    print(df_feeds)
    
    feeds_html = feeds_to_html(df_feeds)
    print(feeds_html)