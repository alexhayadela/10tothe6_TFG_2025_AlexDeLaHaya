from db.base import supabase_client


def get_news(date):
    supabase = supabase_client()
    query = (supabase
             .table("news")
             .select("title,body,url")
             .eq("date",date)
             .order("relevance", desc=True)
             .limit(10)
    )
    res = query.execute()
    
    return res.data

if __name__ == "__main__":
    news = get_news("2026-02-13")
    print(news)