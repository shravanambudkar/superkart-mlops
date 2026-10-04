"""Step 3 - Tune, evaluate and register the best sales-forecasting model on the HF model hub."""
import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

DATA_REPO = "shravanda/superkart-sales-data"
MODEL_REPO = "shravanda/superkart-sales-model"
MODEL_FILE = "best_superkart_model_v1.joblib"
OUT_DIR = "superkart_project/model_building"

api = HfApi(token=os.getenv("HF_TOKEN"))

# ---- 1. Load the train / test splits from the HF dataset repo
base = f"hf://datasets/{DATA_REPO}"
Xtrain = pd.read_csv(f"{base}/Xtrain.csv")
Xtest = pd.read_csv(f"{base}/Xtest.csv")
ytrain = pd.read_csv(f"{base}/ytrain.csv").squeeze("columns")
ytest = pd.read_csv(f"{base}/ytest.csv").squeeze("columns")
print(f"Train: {Xtrain.shape}  Test: {Xtest.shape}")

# ---- 2. Preprocessing: scale numeric features, one-hot encode categorical features
numeric_features = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
categorical_features = ["Product_Sugar_Content", "Product_Type", "Store_Size", "Store_Location_City_Type", "Store_Type"]
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# ---- 3. Define candidate models and their hyper-parameter grids
candidates = {
    "RandomForest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [8, 12, None],
            "model__min_samples_leaf": [1, 4],
            "model__max_features": [0.5, 1.0],
        },
    ),
    "XGBoost": (
        xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1),
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
    ),
}


def evaluate(model, X, y):
    """Regression metrics used to judge forecast quality."""
    pred = model.predict(X)
    n, p = X.shape
    r2 = r2_score(y, pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2),
        "Adj_R2": float(1 - (1 - r2) * (n - 1) / (n - p - 1)),
        "MAPE": float(mean_absolute_percentage_error(y, pred)),
    }


# ---- 4. Tune every candidate with 5-fold cross-validation and evaluate it
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results, fitted = {}, {}
for name, (estimator, grid) in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    search.fit(Xtrain, ytrain)
    fitted[name] = search.best_estimator_
    results[name] = {
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "cv_rmse": float(-search.best_score_),
        "train": evaluate(search.best_estimator_, Xtrain, ytrain),
        "test": evaluate(search.best_estimator_, Xtest, ytest),
    }
    print(f"\n=== {name} ===")
    print("Best params:", results[name]["best_params"])
    print(f"CV RMSE: {results[name]['cv_rmse']:.2f}")
    print(pd.DataFrame({"train": results[name]["train"], "test": results[name]["test"]}).round(4).T.to_string())

# ---- 5. Select the best model on cross-validated RMSE (test set is for reporting only)
best_name = min(results, key=lambda k: results[k]["cv_rmse"])
best_model = fitted[best_name]
print(f"\nBest model: {best_name} (CV RMSE = {results[best_name]['cv_rmse']:.2f}, "
      f"test R2 = {results[best_name]['test']['R2']:.4f})")

# ---- 6. Save the model + metrics locally
os.makedirs(OUT_DIR, exist_ok=True)
model_path = os.path.join(OUT_DIR, MODEL_FILE)
joblib.dump(best_model, model_path)
metrics = {"best_model": best_name, "results": results}
metrics_path = os.path.join(OUT_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

# ---- 7. Register the best model in the HF model hub
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' already exists - reusing it.")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, token=os.getenv("HF_TOKEN"))
    print(f"Model repo '{MODEL_REPO}' created.")

t = results[best_name]["test"]
card = f"""---
license: mit
library_name: sklearn
tags: [tabular-regression, sales-forecasting, {best_name.lower()}]
datasets: [{DATA_REPO}]
---
# SuperKart Sales Forecasting Model

Predicts `Product_Store_Sales_Total` (revenue of a product in a store) for SuperKart.
Serialized scikit-learn `Pipeline` (preprocessing + **{best_name}**) tuned with 5-fold GridSearchCV.

| Metric (test set) | Value |
|---|---|
| RMSE | {t['RMSE']:.2f} |
| MAE | {t['MAE']:.2f} |
| R2 | {t['R2']:.4f} |
| MAPE | {t['MAPE']:.4f} |

Best hyper-parameters: `{results[best_name]['best_params']}`

```python
import joblib
from huggingface_hub import hf_hub_download
model = joblib.load(hf_hub_download("{MODEL_REPO}", "{MODEL_FILE}"))
```
"""
card_path = os.path.join(OUT_DIR, "MODEL_CARD.md")
with open(card_path, "w") as f:
    f.write(card)

for local, remote in [(model_path, MODEL_FILE), (metrics_path, "model_metrics.json"), (card_path, "README.md")]:
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=MODEL_REPO, repo_type="model",
                    commit_message=f"Register {best_name} model artefact: {remote}")
print(f"Registered best model -> https://huggingface.co/{MODEL_REPO}")
