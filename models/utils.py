from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "artifacts"
DB_PATH.mkdir(exist_ok=True)


def get_artifacts_path():
    return DB_PATH
