import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json
import time


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


def query_llm(system_prompt, user_prompt):
    client = create_llm_client()
    model = get_default_model()

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    output_text = response.output_text
    data = json.loads(output_text)


    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return {
        "data": data,
        "usage": usage,
    }

