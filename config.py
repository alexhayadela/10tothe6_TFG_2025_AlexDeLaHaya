from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

# files
ENV_PATH = BASE_DIR / ".env"

# directories
DATA_PATH = BASE_DIR / "data"
ARTIFACTS_PATH = BASE_DIR / "artifacts"

DATA_PATH.mkdir(exist_ok=True)
ARTIFACTS_PATH.mkdir(exist_ok=True)


def load_env() -> None:
    """Load environment variables."""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

