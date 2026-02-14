import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json
from llm.rate_limit import RateLimitState


def load_env():
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)


def create_llm_client() -> OpenAI:
    load_env()

    return OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
   

def get_default_model() -> str:
    return "openai/gpt-oss-120b"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


rate_limit = RateLimitState()
client = create_llm_client()
model = get_default_model()  

def query_llm(system_prompt: str, user_prompt: str, estimated_tokens: int = None):
    """Handles automatic rate-limiting and returns parsed JSON output."""
    if estimated_tokens is None:
        estimated_tokens = 2*estimate_tokens(system_prompt + user_prompt)

    # ---- BLOCK until request can be made ----
    rate_limit.wait_for_slot(estimated_tokens)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    # record tokens & request
    total_tokens = response.usage.total_tokens
    rate_limit.record(total_tokens)

    # parse JSON output
    data = json.loads(response.output_text)

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": total_tokens,
    }

    print("Estimated tokens:", estimated_tokens)
    print(usage)

    return {"data": data, "usage": usage}
