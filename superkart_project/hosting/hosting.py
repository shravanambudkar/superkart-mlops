"""Step 4 - Push the deployment folder (Dockerfile, app.py, requirements.txt, README.md) to the HF Space."""
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

SPACE_ID = "shravanda/SuperKart-Sales-Forecast"
api = HfApi(token=os.getenv("HF_TOKEN"))

# Create a public Docker Space if it does not exist yet
try:
    api.repo_info(repo_id=SPACE_ID, repo_type="space")
    print(f"Space '{SPACE_ID}' already exists - updating it.")
except RepositoryNotFoundError:
    create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="docker", private=False, token=os.getenv("HF_TOKEN"))
    print(f"Space '{SPACE_ID}' created.")

api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id=SPACE_ID,
    repo_type="space",
    path_in_repo="",
    commit_message="Deploy SuperKart Streamlit app",
)
print(f"Deployed -> https://huggingface.co/spaces/{SPACE_ID}")
