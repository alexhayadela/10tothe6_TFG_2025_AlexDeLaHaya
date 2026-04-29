import json
import pytest
import unittest.mock as mock
from llm.gpt_service import LLMService


@pytest.fixture
def llm_service():
    with mock.patch("llm.gpt_service.create_llm_client"):
        svc = LLMService()
    return svc


def _make_response(data: list) -> mock.MagicMock:
    resp = mock.MagicMock()
    resp.output_text = json.dumps(data)
    resp.usage.total_tokens = 100
    resp.usage.input_tokens = 60
    resp.usage.output_tokens = 40
    return resp


def test_query_returns_data_and_usage(llm_service):
    payload = [{"category": "macro_economic", "companies": [], "sentiment": "neutral"}]
    llm_service._client = mock.MagicMock()
    llm_service._client.responses.create.return_value = _make_response(payload)
    result = llm_service.query("sys", "user")
    assert result["data"] == payload
    assert result["usage"]["total_tokens"] == 100


def test_query_records_token_usage(llm_service):
    payload = [{"category": "generic_noise", "companies": [], "sentiment": "neutral"}]
    llm_service._client = mock.MagicMock()
    llm_service._client.responses.create.return_value = _make_response(payload)
    llm_service.query("sys", "user")
    assert len(llm_service._rate_limit.token_events) == 1


def test_estimate_tokens_proportional(llm_service):
    text = "a" * 40
    assert llm_service.estimate_tokens(text) == 10


def test_estimate_tokens_min_one(llm_service):
    assert llm_service.estimate_tokens("") == 1
