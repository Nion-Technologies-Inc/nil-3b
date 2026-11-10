import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# Disable Xet transfer protocol to use the HTTP/LFS upload path.
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "nil-3b-q4_k_m.gguf"
README_PATH = ROOT / "README.md"
REPO_ID = os.getenv("HF_REPO_ID", "nioncorp/nil-3b-gguf").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()


def main() -> None:
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is missing. Set it in the repository .env file or environment."
        )
    if not REPO_ID:
        raise ValueError("HF_REPO_ID must not be empty.")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if MODEL_PATH.stat().st_size == 0:
        raise ValueError(f"Model file is empty: {MODEL_PATH}")
    if not README_PATH.is_file():
        raise FileNotFoundError(f"Model card not found: {README_PATH}")

    api = HfApi(token=HF_TOKEN)
    print(f"Creating Hugging Face model repository if needed: {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    for path in (README_PATH, MODEL_PATH):
        print(f"Uploading {path.name}; the model transfer may take a while.")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=REPO_ID,
            repo_type="model",
        )
    print(f"Upload complete: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
