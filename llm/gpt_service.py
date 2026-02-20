import os
import json
from pathlib import Path
from openai import OpenAI

from llm.rate_limit import RateLimitState


def create_llm_client() -> OpenAI:
    """Returns an llm client."""
    return OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1")
   

def get_default_model() -> str:
    """Returns default llm model."""
    return "openai/gpt-oss-120b"


class LLMService:
    def __init__(self):
        self._client = create_llm_client()
        self._model = get_default_model()
        self._rate_limit = RateLimitState()

    def estimate_tokens(self, text: str) -> int:
        """Estimates the number of tokens the llm call will consume."""
        return max(1, len(text) // 4)

    def query(self,system_prompt: str,user_prompt: str) -> dict:
        """Handles automatic rate-limiting and returns parsed JSON output."""
        estimated_tokens = 2 * self.estimate_tokens(system_prompt + user_prompt)

        self._rate_limit.wait_for_slot(estimated_tokens)

        response = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        total_tokens = response.usage.total_tokens
        self._rate_limit.record(total_tokens)

        data = json.loads(response.output_text)

        return {
            "data": data,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": total_tokens,
            },
        }