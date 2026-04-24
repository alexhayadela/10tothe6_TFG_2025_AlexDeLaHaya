import datetime
import pytest
import unittest.mock as mock
from news.classification import (
    extract_keywords_hit,
    compute_relevance,
    split_into_batches,
    build_news_batch_prompt,
    news_classifier_prompt,
    classify_news,
    IMPORTANT_KEYWORDS,
)


def test_extract_keywords_hit_match():
    result = extract_keywords_hit("los resultados fueron buenos", IMPORTANT_KEYWORDS)
    assert "resultados" in result


def test_extract_keywords_hit_no_match():
    result = extract_keywords_hit("noticia sin palabras clave", IMPORTANT_KEYWORDS)
    assert result == []


def test_extract_keywords_hit_case_insensitive():
    result = extract_keywords_hit("DIVIDENDO anunciado hoy", IMPORTANT_KEYWORDS)
    assert "dividendo" in result


def test_compute_relevance_company_specific_max():
    score = compute_relevance("company_specific", ["BBVA.MC"], "positive", ["dividendo"])
    assert score == pytest.approx(min(0.4 + 0.2 + 0.2 + 0.1, 1.0), abs=1e-3)


def test_compute_relevance_generic_noise_low():
    score = compute_relevance("generic_noise", [], "neutral", [])
    assert score == pytest.approx(0.0)


def test_compute_relevance_with_companies_adds_points():
    base  = compute_relevance("macro_economic", [], "neutral", [])
    with_ = compute_relevance("macro_economic", ["IBEX"], "neutral", [])
    assert with_ > base


def test_compute_relevance_clamped():
    score = compute_relevance("company_specific", ["A"], "positive", list(IMPORTANT_KEYWORDS))
    assert 0.0 <= score <= 1.0


def test_split_into_batches_correct_sizes():
    items = [{"id": i} for i in range(25)]
    batches = split_into_batches(items, batch_size=10)
    assert len(batches) == 3
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5


def test_split_into_batches_empty():
    assert split_into_batches([], batch_size=10) == []


def test_build_news_batch_prompt_contains_titles():
    batch = [{"title": "Banco Santander sube", "body": "Descripcion"}]
    prompt = build_news_batch_prompt(batch)
    assert "Banco Santander sube" in prompt


def test_build_news_batch_prompt_ids_start_at_1():
    batch = [{"title": "T1", "body": "B1"}, {"title": "T2", "body": "B2"}]
    prompt = build_news_batch_prompt(batch)
    assert "id: 1" in prompt
    assert "id: 2" in prompt


def test_news_classifier_prompt_returns_string():
    p = news_classifier_prompt()
    assert isinstance(p, str) and len(p) > 10


def test_classify_news_calls_llm_once_per_batch():
    news = [{"title": f"T{i}", "body": f"B{i}"} for i in range(15)]
    llm_output = [{"category": "generic_noise", "companies": [], "sentiment": "neutral"}]

    with mock.patch("news.classification.LLMService") as MockLLM:
        instance = MockLLM.return_value
        instance.query.side_effect = [
            {"data": llm_output * 10, "usage": {}},
            {"data": llm_output * 5,  "usage": {}},
        ]
        result = classify_news(news)

    assert instance.query.call_count == 2
    assert len(result) == 15


def test_classify_news_output_has_required_keys():
    news = [{"title": "T1", "body": "B1", "url": "http://x.com", "date": "2024-01-01",
             "source": "exp", "section": "mercados", "tags": []}]
    llm_out = [{"category": "company_specific", "companies": ["BBVA.MC"], "sentiment": "positive"}]
    with mock.patch("news.classification.LLMService") as MockLLM:
        MockLLM.return_value.query.return_value = {"data": llm_out, "usage": {}}
        result = classify_news(news)
    item = result[0]
    for key in ("category", "companies", "sentiment", "relevance"):
        assert key in item
