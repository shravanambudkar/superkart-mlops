# SuperKart Sales Forecasting — MLOps Pipeline

End-to-end MLOps pipeline that forecasts `Product_Store_Sales_Total` for SuperKart and ships it as a Streamlit app.

| | |
|---|---|
| HF Space (frontend) | https://huggingface.co/spaces/shravanda/SuperKart-Sales-Forecast |
| HF Dataset | https://huggingface.co/datasets/shravanda/superkart-sales-data |
| HF Model | https://huggingface.co/shravanda/superkart-sales-model |

```
.github/workflows/pipeline.yml      CI/CD: register ► prep ► train ► deploy ► update-main
superkart_project/
├── data/SuperKart.csv              raw data
├── model_building/
│   ├── data_register.py            register raw data on the HF dataset hub
│   ├── prep.py                     clean + split, push splits to HF
│   ├── train.py                    GridSearchCV (RF, XGBoost), evaluate, register best model
│   └── model_metrics.json          latest metrics (auto-updated by CI)
├── deployment/                     Dockerfile, app.py, requirements.txt, README.md (HF Space)
├── hosting/hosting.py              push deployment files to the HF Space
└── requirements.txt                CI dependencies
SuperKart_MLOps_Pipeline.ipynb      project notebook
```

The workflow runs on every push to `main`. It needs a repository secret **`HF_TOKEN`** (Hugging Face write token).
