"""Step 1 - Register the raw SuperKart dataset on the Hugging Face Hub (dataset repo)."""
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

REPO_ID = "shravanda/superkart-sales-data"
REPO_TYPE = "dataset"
RAW_FILE = "superkart_project/data/SuperKart.csv"

api = HfApi(token=os.getenv("HF_TOKEN"))  # token from env (GitHub secret) or cached login

# Create the dataset repo only if it does not exist yet (idempotent for CI re-runs)
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Dataset repo '{REPO_ID}' already exists - reusing it.")
except RepositoryNotFoundError:
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, token=os.getenv("HF_TOKEN"))
    print(f"Dataset repo '{REPO_ID}' created.")

# Upload the raw csv to the root of the dataset repo
api.upload_file(
    path_or_fileobj=RAW_FILE,
    path_in_repo="SuperKart.csv",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    commit_message="Register raw SuperKart data",
)
print(f"Uploaded {RAW_FILE} -> https://huggingface.co/datasets/{REPO_ID}")
