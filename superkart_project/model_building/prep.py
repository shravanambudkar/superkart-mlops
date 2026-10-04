"""Step 2 - Load raw data from the HF Hub, clean it, split it and push the splits back to the Hub."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

REPO_ID = "shravanda/superkart-sales-data"
DATASET_PATH = f"hf://datasets/{REPO_ID}/SuperKart.csv"
OUT_DIR = "superkart_project/data"
TARGET = "Product_Store_Sales_Total"

api = HfApi(token=os.getenv("HF_TOKEN"))

# ---- Load directly from the Hugging Face dataset repo
df = pd.read_csv(DATASET_PATH)
print("Loaded from HF Hub:", df.shape)

# ---- Cleaning
df = df.drop_duplicates()
# Harmonise the inconsistent sugar-content label ('reg' is the same as 'Regular')
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
# Drop identifier columns: Product_Id is unique per row, Store_Id is fully described by the store attributes
df = df.drop(columns=["Product_Id", "Store_Id"])
df = df.dropna(subset=[TARGET])
print("After cleaning:", df.shape)
print("Sugar content levels:", sorted(df["Product_Sugar_Content"].unique()))

# ---- Split into features / target and train / test
X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Save locally
os.makedirs(OUT_DIR, exist_ok=True)
splits = {"Xtrain.csv": Xtrain, "Xtest.csv": Xtest, "ytrain.csv": ytrain, "ytest.csv": ytest}
for name, frame in splits.items():
    frame.to_csv(os.path.join(OUT_DIR, name), index=False)
print(f"Train: {Xtrain.shape}  Test: {Xtest.shape}")

# ---- Upload the splits back to the HF dataset repo
for name in splits:
    api.upload_file(
        path_or_fileobj=os.path.join(OUT_DIR, name),
        path_in_repo=name,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Upload {name}",
    )
    print(f"Uploaded {name}")
