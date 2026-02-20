from news.news_rss import last_news
from llm.gpt_service import LLMService


IMPORTANT_KEYWORDS = {
    "resultados",
    "opa",
    "dividendo",
    "beneficio",
    "guidance",
    "adquisición"
}

CATEGORY_WEIGHTS = {
    "company_specific": 0.4,
    "macro_economic": 0.3,
    "market_sentiment": 0.2,
    "generic_noise": 0.0
}


def extract_keywords_hit(text: str, keywords: set[str]) -> list[str]:
    """Extracts keywords."""
    text = text.lower()
    return [k for k in keywords if k in text]


def compute_relevance(category: str,companies: list[str],sentiment: str,keywords_hit: list[str]) -> float:
    """Computes relevance of news."""
    score = 0.0
    score += CATEGORY_WEIGHTS.get(category, 0.0)

    if companies:
        score += 0.2

    keywords_hit = {k.lower() for k in keywords_hit}
    if keywords_hit & IMPORTANT_KEYWORDS:
        score += 0.2

    # urgency, not direction
    if sentiment != "neutral":
        score += 0.1

    return round(min(score, 1.0), 3)


def news_classifier_prompt():
    "Returns prompt for news classification."
    prompt ="""
    You are a financial news classifier.

    For EACH news item, classify it into ONE category:

    - company_specific
    - macro_economic
    - market_sentiment
    - generic_noise

    Also extract:
    - companies mentioned
    - sentiment: positive, negative, neutral

    Respond strictly in JSON.
    The output MUST be a list with the same length and order as the input.

    JSON format:
    [
    {
        "category": "...",
        "companies": [],
        "sentiment": "..."
    }
    ]"""
    return prompt


def build_news_batch_prompt(news_batch: list[dict]) -> str:
    """Builds structured LLM input for a batch of news items."""
    return "\n".join(
        f"id: {i}\n"
        f"title: {item['title']}\n"
        f"body: {item['body']}\n"
        for i, item in enumerate(news_batch, start=1)
    )


def split_into_batches(items: list[dict],batch_size: int = 10) -> list[list[dict]]:
    """Splits news items into batches."""
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def classify_news(news: list[dict]) -> list[dict]:
    """Classifies news items."""
    llm = LLMService()
    batches = split_into_batches(news, batch_size=10)

    all_outputs = []

    system_prompt = news_classifier_prompt()
    for batch in batches:
        user_prompt = build_news_batch_prompt(batch)
        output = llm.query(system_prompt,user_prompt)
        all_outputs.extend(output["data"])

    if len(all_outputs) != len(news):
        raise ValueError("LLM output missmatch")

    scored_news = []

    for feed, llm_out in zip(news, all_outputs):
        keywords_hit = extract_keywords_hit(
            feed["title"] + " " + feed["body"],
            IMPORTANT_KEYWORDS
        )

        relevance = compute_relevance(
            category=llm_out["category"],
            companies=llm_out["companies"],
            sentiment=llm_out["sentiment"],
            keywords_hit=keywords_hit
        )

        scored_news.append({
            **feed,
            **llm_out,
            "relevance": relevance
        })

    return scored_news


if __name__ == "__main__":

    news = last_news()
    classified_news = classify_news(news)
    """
    top_news = sorted(
    classified_news,
    key=lambda x: x["relevance"],
    reverse=True
    )[:10]
    """
    for new in classified_news:
        print(new)
