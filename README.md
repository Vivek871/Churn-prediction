# 🔮 Customer Churn Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)

**A production-grade ML system that predicts customer churn with 84% AUC — from raw data to deployed API.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Results](#-model-results) • [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

This project builds a complete, production-ready **Customer Churn Prediction System** on the IBM Telco dataset. It follows real-world ML engineering practices — not just model training, but the full lifecycle from data validation to containerized deployment.

> **Business Problem:** 26.5% of telecom customers churn every month. Identifying at-risk customers early allows the retention team to intervene before it's too late.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **EDA Notebooks** | Full exploratory analysis with visual insights |
| 🧪 **Feature Engineering** | 6 domain-driven features backed by data evidence |
| 🤖 **Model Benchmarking** | Compared LogReg, RandomForest, XGBoost fairly |
| ⚖️ **Class Imbalance Handling** | SMOTE via ImbPipeline (no data leakage) |
| 🎯 **Threshold Tuning** | Optimal F1 threshold search (not default 0.5) |
| 📊 **MLflow Tracking** | Every experiment logged, compared, versioned |
| 🗂️ **Model Registry** | Staging → Production promotion workflow |
| 🚀 **FastAPI** | Production REST API with Pydantic validation |
| 🐳 **Docker** | Fully containerized, runs anywhere |
| 🔄 **CI/CD** | GitHub Actions — lint, test, build on every push |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                               │
│                                                                 │
│  Raw CSV → basic_clean() → validate() → create_domain_features()│
│                                                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE                                  │
│                                                                 │
│  ColumnTransformer → SMOTE → XGBClassifier                      │
│  (ImbPipeline — SMOTE only on train folds, never test)          │
│                                                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MLFLOW EXPERIMENT TRACKING                      │
│                                                                 │
│  Log params → Log metrics → Register model → Promote to Staging │
│                                                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI INFERENCE                             │
│                                                                 │
│  POST /predict → Pydantic validation → Feature engineering      │
│               → Load Staging model → Return risk level          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- `uv` package manager

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/churn-production.git
cd churn-production
```

### 2. Setup environment
```bash
uv venv .venv --python python3.11
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
```

### 3. Download dataset
```bash
mkdir -p data/raw
wget -O data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
```

### 4. Train the model
```bash
python src/training/train.py
```

### 5. Start with Docker Compose
```bash
docker build -t churn-api:latest .
docker-compose up
```

### 6. Test the API
```bash
curl http://localhost:8000/health
```

---

## 📡 API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### `GET /health`
Health check — returns model status and version.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "2",
  "threshold": 0.33
}
```

---

#### `POST /predict`
Predict churn probability for a single customer.

**Request body:**
```json
{
  "tenure": 2,
  "MonthlyCharges": 85.0,
  "TotalCharges": 170.0,
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "PaperlessBilling": "Yes",
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No"
}
```

**Response:**
```json
{
  "churn_probability": 0.8621,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "threshold_used": 0.33,
  "model_version": "2"
}
```

**Risk levels:**
| Level | Probability | Action |
|---|---|---|
| 🟢 LOW | < 0.40 | No action needed |
| 🟡 MEDIUM | 0.40 – 0.70 | Monitor closely |
| 🔴 HIGH | > 0.70 | Immediate retention outreach |

#### `GET /docs`
Interactive Swagger UI — test all endpoints in browser.

---

## 📊 Model Results

### Benchmark Comparison
| Model | CV AUC | Test AUC | Test F1 |
|---|---|---|---|
| **XGBoost** ✅ | 0.8443 | 0.8419 | 0.6320 |
| LogisticRegression | 0.8477 | 0.8411 | 0.6259 |
| RandomForest | 0.8428 | 0.8379 | 0.6272 |

### Final Model Performance
```
ROC-AUC:          0.8387
F1 Score:         0.6263
Recall:           0.7754   ← catches 77.5% of all churners
Precision:        0.5254
Optimal Threshold: 0.33   ← tuned from default 0.5
```

### Key EDA Findings
```
Contract type gap:    42.7% vs 2.8%   churn rate  (40pt difference)
Tenure gap:           47.4% vs 9.6%   churn rate  (38pt difference)
Internet service gap: 41.9% vs 7.4%   churn rate  (35pt difference)
Payment method gap:   45.3% vs 15.0%  churn rate  (30pt difference)
```

### Engineered Features
| Feature | Correlation | Business Reason |
|---|---|---|
| `is_month_to_month` | +0.405 | No commitment = easy to leave (6.3x lift) |
| `charges_per_tenure` | +0.412 | High spend + new customer = risk |
| `no_support_services` | +0.383 | Unsupported customers leave (3.6x lift) |
| `is_electronic_check` | +0.302 | Manual payment friction (2.7x lift) |
| `tenure_band` | — | Lifecycle cohort signal |
| `service_count` | -0.086 | More services = more engaged |

---

## 📁 Project Structure

```
churn-production/
│
├── 📓 notebooks/
│   ├── 01_eda.ipynb                   # data exploration + insights
│   ├── 02_feature_engineering.ipynb   # feature analysis + correlation
│   └── 03_model_experiments.ipynb     # model benchmarking + importance
│
├── 🐍 src/
│   ├── data/
│   │   ├── loader.py                  # data loading + basic cleaning
│   │   └── validator.py               # schema + data quality checks
│   ├── features/
│   │   ├── engineer.py                # domain feature creation
│   │   └── preprocessor.py            # ColumnTransformer pipeline
│   ├── models/
│   │   ├── pipeline.py                # ImbPipeline (SMOTE + XGBoost)
│   │   ├── evaluator.py               # metrics + threshold tuning
│   │   ├── benchmark.py               # model comparison
│   │   └── registry.py                # MLflow model registry
│   ├── training/
│   │   └── train.py                   # full training orchestration
│   ├── api/
│   │   ├── app.py                     # FastAPI app + endpoints
│   │   ├── model.py                   # model loading + prediction
│   │   └── schemas.py                 # Pydantic request/response
│   └── utils/
│       ├── logger.py                  # structured logging
│       └── config.py                  # yaml + .env config loader
│
├── ⚙️ configs/
│   └── config.yaml                    # model + feature config
│
├── 🧪 tests/
│   └── test_api.py                    # 10 unit tests
│
├── 🐳 Dockerfile                      # container definition
├── 🐳 docker-compose.yml              # multi-service orchestration
├── 🔄 .github/workflows/ci.yml        # CI/CD pipeline
├── 📋 requirements.txt
└── 🚀 main.py                         # uvicorn entry point
```

---

## 🔄 CI/CD Pipeline

Every push to `main` or `develop` triggers:

```
Push to GitHub
     │
     ├── ✅ Install dependencies
     ├── ✅ Download dataset
     ├── ✅ Lint with ruff
     ├── ✅ Run 10 pytest tests
     │
     └── (if tests pass)
         ├── ✅ Build Docker image
         └── ✅ Verify image
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML Framework | Scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Experiment Tracking | MLflow 2.14 |
| API Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Package Manager | uv |
| Logging | Python logging (structured) |
| Config Management | YAML + python-dotenv |

---

## 📈 MLflow Tracking

Start the MLflow UI:
```bash
mlflow ui --backend-store-uri mlruns/ --port 5000
```

Open: **http://localhost:5000**

Every training run logs:
- All hyperparameters
- CV metrics (mean ± std)
- Test metrics
- Model artifact with signature
- Config file
- Optimal threshold

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

```
tests/test_api.py::test_load_raw_data                    PASSED
tests/test_api.py::test_basic_clean                      PASSED
tests/test_api.py::test_validate_passes_on_clean_data    PASSED
tests/test_api.py::test_validate_fails_on_bad_data       PASSED
tests/test_api.py::test_domain_features_created          PASSED
tests/test_api.py::test_charges_per_tenure_no_division   PASSED
tests/test_api.py::test_is_month_to_month_binary         PASSED
tests/test_api.py::test_service_count_range              PASSED
tests/test_api.py::test_churn_request_valid              PASSED
tests/test_api.py::test_churn_request_invalid_tenure     PASSED

10 passed in 0.42s
```

---

## 🔑 Key Production Patterns

```
✅ Fail fast validation    — catch bad data at entry point, not inside model
✅ ImbPipeline             — SMOTE only on train folds, never test (no leakage)
✅ Threshold tuning        — optimal F1 threshold, not default 0.5
✅ MLflow signatures       — input/output schema enforcement at inference
✅ Model fallback          — MLflow → local joblib if registry unavailable
✅ Structured logging      — timestamp, level, file:line on every log
✅ Config separation       — secrets in .env, hyperparams in config.yaml
✅ Docker volumes          — models updated without container rebuild
✅ Health endpoint         — infrastructure-ready liveness check
```

---

## 👤 Author

**Mrityunjay**
- GitHub: [@mrityunjay835](https://github.com/mrityunjay835)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**Built step by step — every decision backed by data.**

⭐ Star this repo if you found it useful!

</div>