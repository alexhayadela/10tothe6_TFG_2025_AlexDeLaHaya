from news.news_rss import last_news
from llm.gpt_service import query_llm, estimate_tokens
from llm.rate_limit_state import RateLimitState
import time

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


def news_classifier_prompt():
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
]
"""
    return prompt


def build_news_batch_prompt(news_batch: list[dict]) -> str:
    """
    Builds structured LLM input for a batch of news items.
    """
    return "\n".join(
        f"id: {i}\n"
        f"title: {item['title']}\n"
        f"summary: {item['summary']}\n"
        for i, item in enumerate(news_batch, start=1)
    )


def split_into_batches(
    items: list[dict],
    batch_size: int = 10
) -> list[list[dict]]:
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def extract_keywords_hit(text: str, keywords: set[str]) -> list[str]:
    text = text.lower()
    return [k for k in keywords if k in text]


def compute_relevance_score(
    category: str,
    companies: list[str],
    sentiment: str,
    keywords_hit: list[str]
) -> float:
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


def classify_news(news: list[dict]) -> list[dict]:
    
    batches = split_into_batches(news, batch_size=10)

    all_outputs = []

    system_prompt = news_classifier_prompt()
    est_system = estimate_tokens(system_prompt)
    rate_limit = RateLimitState()
    for batch in batches:
        user_prompt = build_news_batch_prompt(batch)
        est_user = estimate_tokens(user_prompt)
        estimated_tokens = (est_system + est_user)*4/3
        sleep_time = rate_limit.can_make_request(estimated_tokens)
        if sleep_time > 0:
            print(f"⏳ Sleeping {sleep_time:.2f}s to respect rate limits")
            time.sleep(sleep_time)
        output = query_llm(system_prompt,user_prompt)
        print(output["usage"])
        rate_limit.record(int(output["usage"]["total_tokens"]))
        all_outputs.extend(output["data"])

    assert len(all_outputs) == len(news)

    scored_news = []

    for feed, llm_out in zip(news, all_outputs):
        keywords_hit = extract_keywords_hit(
            feed["title"] + " " + feed["summary"],
            IMPORTANT_KEYWORDS
        )

        relevance = compute_relevance_score(
            category=llm_out["category"],
            companies=llm_out["companies"],
            sentiment=llm_out["sentiment"],
            keywords_hit=keywords_hit
        )

        scored_news.append({
            **feed,
            **llm_out,
            "relevance_score": relevance
        })

    return scored_news


# Test    
if __name__ == "__main__":

    news = last_news()
    classified_news = classify_news(news)
    """
    top_news = sorted(
    classified_news,
    key=lambda x: x["relevance_score"],
    reverse=True
)[:10]
    """
    for new in classified_news:
        print(new)
