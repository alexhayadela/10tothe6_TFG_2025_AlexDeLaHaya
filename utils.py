from dotenv import load_dotenv
from pathlib import Path


def load_env() -> None:
    """Load environment variables."""
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)




