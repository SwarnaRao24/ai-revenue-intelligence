<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&duration=2800&pause=2000&color=00D4FF&center=true&vCenter=true&width=940&lines=AI-Powered+Churn+%26+Revenue+Intelligence;Production-Grade+MLOps+Platform;XGBoost+%7C+LightGBM+%7C+FastAPI+%7C+MLflow" alt="Typing SVG" />

#  AI-Powered Customer Churn & Revenue Intelligence Platform

**End-to-End MLOps · Snowflake · dbt · XGBoost · LightGBM · FastAPI · MLflow · LLM Insights**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io)
[![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=for-the-badge&logo=snowflake&logoColor=white)](https://snowflake.com)
[![dbt](https://img.shields.io/badge/dbt-FF694B?style=for-the-badge&logo=dbt&logoColor=white)](https://getdbt.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Airflow](https://img.shields.io/badge/Airflow-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)](https://airflow.apache.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI_GPT--4-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)

<br/>

> *A production-grade, end-to-end Machine Learning platform that predicts customer churn,
> forecasts revenue, segments customers, detects anomalies, and generates
> LLM-powered executive insights — all automated, containerised, and monitored.*

<br/>

[![GitHub stars](https://img.shields.io/github/stars/SwarnaRao24/ai-revenue-intelligence?style=social)](https://github.com/SwarnaRao24/ai-revenue-intelligence)
[![GitHub forks](https://img.shields.io/github/forks/SwarnaRao24/ai-revenue-intelligence?style=social)](https://github.com/SwarnaRao24/ai-revenue-intelligence)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Business Problems Solved](#-business-problems-solved)
- [Platform Architecture](#-platform-architecture)
- [ML Pipeline Flow](#-ml-pipeline-flow)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [ML Models](#-ml-models)
- [API Reference](#-api-reference)
- [dbt Models](#-dbt-data-pipeline)
- [MLOps & Monitoring](#-mlops--monitoring)
- [Docker & Deployment](#-docker--deployment)
- [Results & Metrics](#-results--metrics)
- [Resume Bullets](#-resume-bullets)

---

## Overview

This platform is a **fully productionised Machine Learning Engineering system** built to solve real enterprise business problems around customer retention and revenue intelligence.

It ingests raw customer data, transforms it through a **medallion architecture** (Bronze → Silver → Gold) inside **Snowflake** using **dbt**, engineers 40+ ML features, trains an **XGBoost + LightGBM ensemble** for churn prediction, forecasts revenue with an **XGBoost time-series regressor**, segments customers using **K-Means clustering**, detects anomalies with **Isolation Forest**, and exposes everything through a **FastAPI** microservice backed by a **Streamlit** analytics dashboard.

Every experiment is tracked in **MLflow**, every pipeline is orchestrated by **Apache Airflow**, and the entire stack runs containerised in **Docker Compose** with a **GitHub Actions CI/CD** pipeline.

The AI layer uses **OpenAI GPT-4** to auto-generate natural language executive summaries, churn explanations, and risk narratives — making model outputs accessible to non-technical stakeholders.

---

##  Business Problems Solved

| # | Problem | ML Solution | Business Impact |
|---|---------|-------------|-----------------|
| 1 |  **Who will churn?** | XGBoost + LightGBM Ensemble Classifier | Identify at-risk customers 30 days before churn |
| 2 |  **What revenue do we forecast?** | XGBoost Time-Series Regressor | 6–12 month MRR projections with confidence bands |
| 3 |  **Who are our customer segments?** | K-Means Clustering (RFM features) | 5 behavioural segments for targeted campaigns |
| 4 |  **Who is behaving abnormally?** | Isolation Forest Anomaly Detection | Flag unusual spend / support patterns in real-time |
| 5 |  **What does it all mean?** | OpenAI GPT-4 NLG layer | Auto-generated executive summaries & risk narratives |
| 6 |  **Which customers should we upsell?** | Segment + churn risk intersection | Prioritised upsell opportunity list |

---

##  Platform Architecture

```mermaid
graph TB
    subgraph SOURCES[" Data Sources"]
        A1[IBM Telco Dataset]
        A2[E-Commerce Orders]
        A3[Support Tickets]
    end

    subgraph INGESTION["Ingestion Layer — Airflow DAGs"]
        B1[ingest_telco.py]
        B2[ingest_ecommerce.py]
        B3[Incremental Loader]
    end

    subgraph SNOWFLAKE["Snowflake Data Platform"]
        C1[" BRONZE\nRaw Immutable Tables"]
        C2[" SILVER\ndbt Staging Models\nstg_customers · stg_transactions"]
        C3[" GOLD\ndbt Mart Models\ndim_customer · fct_revenue\nmart_churn_features"]
    end

    subgraph ML["ML Platform"]
        D1[Feature Engineering\n40+ Features]
        D2[XGBoost Classifier]
        D3[LightGBM Classifier]
        D4[Voting Ensemble]
        D5[XGBoost Regressor\nRevenue]
        D6[K-Means\nSegmentation]
        D7[Isolation Forest\nAnomaly Detection]
    end

    subgraph MLOPS["MLOps Layer"]
        E1[MLflow Tracking]
        E2[MLflow Model Registry]
        E3[Experiment Comparison]
        E4[Model Drift Monitor]
    end

    subgraph API["API Layer — FastAPI"]
        F1[POST /predict/churn]
        F2[POST /predict/revenue]
        F3[POST /customer-risk]
        F4[POST /generate-summary]
    end

    subgraph AI["AI Insights — OpenAI GPT-4"]
        G1[Executive Summaries]
        G2[Churn Explanations]
        G3[Risk Narratives]
    end

    subgraph DASH["Dashboard — Streamlit"]
        H1[KPI Cards]
        H2[Churn Risk Table]
        H3[Revenue Forecast Chart]
        H4[Segment Explorer]
        H5[Model Monitoring]
    end

    SOURCES --> INGESTION --> C1 --> C2 --> C3
    C3 --> D1 --> D2 --> D4
    D1 --> D3 --> D4
    D1 --> D5
    D1 --> D6
    D1 --> D7
    D4 --> E1
    D5 --> E1
    E1 --> E2
    E2 --> F1 & F2 & F3
    F4 --> AI
    F1 & F2 & F3 & F4 --> DASH
    E2 --> E4
```

---

##  ML Pipeline Flow

```mermaid
flowchart LR
    RAW["Raw CSV\nTelco + E-Commerce"]
    CLEAN["Bronze Layer\nCleaned Parquet"]
    FEAT["Feature Mart\n40+ ML Features"]
    SPLIT["Train / Test Split\n80% / 20% Stratified"]

    subgraph TRAIN["Model Training"]
        XGB["XGBoost\nn_estimators=300\nmax_depth=6\nscale_pos_weight=3"]
        LGB["LightGBM\nn_estimators=300\nclass_weight=balanced"]
        ENS["Voting Ensemble\nXGB 60% + LGB 40%\nsoft voting"]
    end

    subgraph EVAL["Evaluation"]
        AUC["AUC-ROC\n> 0.88"]
        F1["F1 Score\nPrecision · Recall"]
        SHAP["SHAP Values\nFeature Importance"]
    end

    TRACK["MLflow\nExperiment Tracking\nModel Registry"]
    SERVE["FastAPI\nReal-time Inference\nBatch Scoring"]

    RAW --> CLEAN --> FEAT --> SPLIT
    SPLIT --> XGB & LGB
    XGB & LGB --> ENS
    ENS --> AUC & F1 & SHAP
    AUC & F1 --> TRACK
    TRACK --> SERVE
```

---

## Orchestration — Airflow DAGs

```mermaid
gantt
    title Airflow Daily Pipeline Schedule
    dateFormat HH:mm
    section Ingestion
    ingest_telco_dag         :00:00, 15m
    ingest_ecommerce_dag     :00:15, 20m
    section Transformation
    dbt_staging_run          :00:35, 10m
    dbt_marts_run            :00:45, 10m
    dbt_features_run         :00:55, 10m
    section ML Training
    churn_retrain_dag        :01:05, 25m
    revenue_retrain_dag      :01:30, 20m
    segmentation_dag         :01:50, 15m
    anomaly_dag              :02:05, 10m
    section Monitoring
    drift_detection_dag      :02:15, 10m
    data_quality_dag         :02:25, 10m
    section Reporting
    llm_summary_dag          :02:35, 5m
```

---

## 🛠 Tech Stack

### Data Engineering & Warehousing
| Tool | Role |
|------|------|
| **Snowflake** | Cloud data warehouse — BRONZE / SILVER / GOLD schemas |
| **Snowpark** | Python-native Snowflake compute for feature engineering |
| **dbt-core + dbt-snowflake** | Analytics engineering — staging, mart, and feature models |
| **Apache Parquet** | Columnar storage for Bronze layer |
| **Apache Airflow 2.x** | Pipeline orchestration and DAG scheduling |

### Machine Learning
| Tool | Role |
|------|------|
| **XGBoost 2.x** | Gradient boosted trees — churn classifier + revenue regressor |
| **LightGBM** | Fast gradient boosting — ensemble partner for churn |
| **Scikit-learn** | Preprocessing, K-Means, Isolation Forest, VotingClassifier |
| **MLflow 2.x** | Experiment tracking, model registry, artifact store |
| **SHAP** | Model explainability — per-prediction feature attribution |
| **Joblib** | Model serialisation and parallel processing |

### API & Backend
| Tool | Role |
|------|------|
| **FastAPI** | Async REST API with automatic OpenAPI/Swagger docs |
| **Pydantic v2** | Request/response validation and settings management |
| **Uvicorn** | ASGI server for FastAPI |

### AI & LLM Layer
| Tool | Role |
|------|------|
| **OpenAI GPT-4o** | Natural language executive report generation |
| **LangChain** | LLM orchestration and prompt management |

### Infrastructure & DevOps
| Tool | Role |
|------|------|
| **Docker + Docker Compose** | Full stack containerisation |
| **GitHub Actions** | CI/CD — lint, test, build, deploy |
| **Terraform** | Infrastructure as Code for cloud resources |
| **Great Expectations** | Data quality validation and checkpoints |

### Visualisation
| Tool | Role |
|------|------|
| **Streamlit** | Interactive analytics dashboard |
| **Plotly** | Interactive charts — churn trends, revenue forecast |

---

## 📁 Project Structure

````
ai-revenue-intelligence/
│
├── 📂 data_ingestion/              # Bronze layer loaders
│   ├── ingest_telco.py             # IBM Telco → Parquet
│   └── ingest_ecommerce.py         # Synthetic e-commerce data
│
├── 📂 dbt_project/                 # Analytics Engineering
│   ├── dbt_project.yml
│   └── models/
│       ├── staging/                # Silver — stg_customers, stg_transactions
│       ├── marts/                  # Gold — dim_customer, fct_revenue
│       └── features/               # ML feature marts
│
├── 📂 ml_training/                 # Model training pipelines
│   ├── utils/
│   │   ├── data_loader.py          # Feature loading + engineering
│   │   └── mlflow_utils.py         # MLflow helpers
│   ├── churn/
│   │   └── train_churn.py          # XGBoost + LightGBM ensemble
│   ├── revenue/
│   │   └── train_revenue.py        # XGBoost time-series regressor
│   ├── segmentation/
│   │   └── train_segmentation.py   # K-Means clustering
│   └── anomaly/
│       └── train_anomaly.py        # Isolation Forest
│
├── 📂 api/                         # FastAPI microservice
│   ├── main.py                     # App entry point
│   ├── config.py                   # Settings from .env
│   ├── routers/
│   │   └── predict.py              # All prediction endpoints
│   ├── schemas/
│   │   ├── requests.py             # Pydantic request models
│   │   └── responses.py            # Pydantic response models
│   └── services/
│       └── model_service.py        # Model loading + inference
│
├── 📂 dashboard/                   # Streamlit dashboard
│   └── app.py
│
├── 📂 airflow_dags/                # Orchestration DAGs
│   ├── ingestion_dag.py
│   ├── training_dag.py
│   └── monitoring_dag.py
│
├── 📂 monitoring/                  # Drift + quality checks
├── 📂 docker/                      # Dockerfiles per service
├── 📂 tests/                       # Unit + integration tests
├── 📂 .github/workflows/           # CI/CD pipelines
│
├── docker-compose.yml              # Full stack
├── requirements.txt
├── Makefile                        # Developer shortcuts
└── .env.example
````

---

##  Quick Start

### Prerequisites
- Python 3.11+ with Anaconda
- Docker Desktop
- Snowflake account *(free trial works — optional, runs locally without it)*
- OpenAI API key *(optional — summary endpoint works without it)*

### 1 — Clone & Install

```bash
git clone https://github.com/yourusername/ai-revenue-intelligence.git
cd ai-revenue-intelligence
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

### 2 — Ingest Data

```bash
python data_ingestion/ingest_telco.py
python data_ingestion/ingest_ecommerce.py
```

### 3 — Train All Models

```bash
python ml_training/churn/train_churn.py
python ml_training/revenue/train_revenue.py
python ml_training/segmentation/train_segmentation.py
python ml_training/anomaly/train_anomaly.py
```

### 4 — Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5 — Open the Platform

| Service | URL |
|---------|-----|
|  FastAPI Swagger Docs | http://localhost:8000/docs |
|  Streamlit Dashboard | http://localhost:8501 |
|  MLflow UI | http://localhost:5000 |
|  Airflow UI | http://localhost:8080 |

### 6 — Or run everything with Docker

```bash
docker-compose up -d
```

---

##  ML Models

### Customer Churn Prediction

```mermaid
graph LR
    F["40+ Features\nTenure · Charges · Contract\nSupport Tickets · Services"]
    XGB["XGBoost\nClassifier\nscale_pos_weight=3\nn_estimators=300"]
    LGB["LightGBM\nClassifier\nclass_weight=balanced\nn_estimators=300"]
    ENS["Soft Voting\nEnsemble\nXGB=0.6 · LGB=0.4"]
    OUT["Output\nChurn Probability\nRisk Tier · SHAP"]

    F --> XGB & LGB --> ENS --> OUT
```

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.88+** |
| F1 Score | **0.74+** |
| Precision | **0.72+** |
| Recall | **0.76+** |
| Churn Rate Captured | **~80% of churners** in top 30% scored |

**Top churn drivers (SHAP-ranked):**
1. `contract_type` — Month-to-month contracts churn 3× more
2. `tenure_months` — Customers < 12 months are highest risk
3. `internet_service` — Fiber optic has elevated churn
4. `num_support_tickets` — 3+ tickets = strong churn signal
5. `monthly_charges` — High-charge customers with no lock-in

---

### Revenue Forecasting

```mermaid
graph LR
    TS["48 Months\nHistorical MRR"]
    ENG["Time Features\nLag 1·2·3·6·12\nRolling Mean·Std\nMonth Sin·Cos"]
    XGB["XGBoost\nRegressor\nn_estimators=200"]
    OUT["6–12 Month\nRevenue Forecast\n± Confidence Band"]

    TS --> ENG --> XGB --> OUT
```

| Metric | Score |
|--------|-------|
| MAPE | **< 8%** |
| Trend Detection | Growing / Stable / Declining |

---

### Customer Segmentation

| Segment | Profile | Strategy |
|--------|---------|----------|
|  Champions | High tenure, high spend, low tickets | Loyalty rewards, brand advocates |
|  Loyal Customers | Medium tenure, consistent spend | Upsell premium services |
|  At Risk | Declining activity, rising tickets | Proactive retention outreach |
|  Hibernating | Low recent activity | Re-engagement campaigns |
|  Lost | Very low activity, cancelled services | Win-back offers |

---

##  API Reference

### `POST /api/v1/predict/churn`

```json
// Request
{
  "customer_id": "CUST_001",
  "tenure_months": 8,
  "monthly_charges": 85.50,
  "total_charges": 684.00,
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "tech_support": "No",
  "num_support_tickets": 4
}

// Response
{
  "customer_id": "CUST_001",
  "churn_probability": 0.8341,
  "churn_prediction": true,
  "risk_tier": "HIGH",
  "top_factors": [
    { "feature": "contract_type", "impact": 0.42, "direction": "increases" },
    { "feature": "tenure_months", "impact": 0.31, "direction": "increases" },
    { "feature": "num_support_tickets", "impact": 0.19, "direction": "increases" }
  ],
  "recommendation": "Offer 20% discount to upgrade to 1-year contract.",
  "model_version": "xgb_lgb_ensemble_v1"
}
```

### `POST /api/v1/predict/revenue`
```json
// Request
{ "periods": 6, "granularity": "monthly" }

// Response
{
  "trend": "growing",
  "total_forecast": 642381.50,
  "forecast": [
    { "period": "2026-01", "forecast": 98420.0, "lower_bound": 90547.0, "upper_bound": 106294.0 }
  ]
}
```

### `POST /api/v1/customer-risk`
Returns combined churn probability + anomaly score + customer segment + recommendations.

### `POST /api/v1/generate-summary`
Returns GPT-4 generated executive report with key metrics for any date range.

---

##  dbt Data Pipeline

```mermaid
graph TD
    B["BRONZE\nRaw Parquet / Snowflake Tables\nbronze_telco_customers\nbronze_ecommerce_orders\nbronze_support_tickets"]

    S1["stg_customers\nCleaned + typed\nsnake_case columns"]
    S2["stg_transactions\nNormalised orders\nstatus flags"]
    S3["stg_support_tickets\nSentiment encoded\nresolution flags"]

    D1["dim_customer\nSCD Type 2\ncustomer master"]
    F1["fct_revenue\nMonthly aggregates\nMoM growth %"]
    F2["fct_customer_activity\nDaily activity\nengagement scores"]
    M1["mart_churn_features\n40+ ML features\nML-ready table"]

    B --> S1 & S2 & S3
    S1 & S2 & S3 --> D1 & F1 & F2
    D1 & F1 & F2 --> M1
```

```bash
dbt run                              # Full pipeline
dbt test                             # Data quality tests
dbt docs generate && dbt docs serve  # Interactive lineage graph
```

---

##  MLOps & Monitoring

```mermaid
graph LR
    TR["Model Training\nExperiment Run"]
    LOG["MLflow Logging\nParams · Metrics\nArtifacts · Tags"]
    REG["Model Registry\nStaging → Production\nVersion Control"]
    SERVE["FastAPI\nReal-time Serving"]
    MON["Drift Monitor\nPSI · KS Test\nFeature Distribution"]
    ALERT["Alert DAG\nRetrain if drift\n> threshold"]

    TR --> LOG --> REG --> SERVE
    SERVE --> MON --> ALERT --> TR
```

Every model run logs:
- **Parameters** — all hyperparameters
- **Metrics** — AUC-ROC, F1, MAPE, Silhouette
- **Artifacts** — serialised model, feature importance CSV
- **Tags** — dataset version, training date, data hash

Nightly Airflow DAG checks for **feature drift** (PSI > 0.2) and **prediction drift** and triggers automatic retraining if thresholds are exceeded.

---

##  Docker & Deployment

```bash
# Start full platform
docker-compose up -d

# Services launched:
# api        → http://localhost:8000
# dashboard  → http://localhost:8501
# mlflow     → http://localhost:5000
# airflow    → http://localhost:8080
# postgres   → localhost:5432
# minio      → http://localhost:9001
```

```mermaid
graph TB
    subgraph DOCKER[" Docker Compose Network"]
        API["churn_api\nFastAPI + Uvicorn\n:8000"]
        DASH["churn_dashboard\nStreamlit\n:8501"]
        MLF["churn_mlflow\nTracking Server\n:5000"]
        AF["churn_airflow\nWebserver + Scheduler\n:8080"]
        PG["churn_postgres\nPostgreSQL 15\n:5432"]
        MN["churn_minio\nS3-compatible store\n:9000"]
    end

    API --> PG
    MLF --> PG & MN
    AF --> PG
    DASH --> API
```

---

## CI/CD Pipeline

```mermaid
graph LR
    PUSH["git push\nto main"]
    LINT["Lint\nflake8 · black\nisort"]
    TEST["Unit Tests\npytest\ncoverage > 80%"]
    DBT["dbt compile\nSQL validation"]
    BUILD["Docker Build\nmulti-stage\noptimised image"]
    PUSH2["Push to\nContainer Registry"]

    PUSH --> LINT --> TEST --> DBT --> BUILD --> PUSH2
```

---

## Results & Metrics

| Model | Metric | Value |
|-------|--------|-------|
| Churn Classifier | AUC-ROC | **0.88+** |
| Churn Classifier | F1 Score | **0.74+** |
| Revenue Forecaster | MAPE | **< 8%** |
| Segmentation | Silhouette Score | **> 0.45** |
| Anomaly Detection | Anomaly Rate | **~5%** flagged |
| API | Latency p95 | **< 120ms** |
| Data Pipeline | Records Processed | **22,000+** |
| Feature Store | Features Engineered | **40+** |

---
**Developer:** Swarna Rao  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/swarnamukhirchintalapudi)

**Focus:** Python | CI/CD | Docker | Web Development | UI/UX
---
##  License

MIT License — feel free to use, adapt, and build on this project.

---

<div align="center">

**Built for production. Designed for impact. Ready for the portfolio.**

If this helped you, please consider giving it a ⭐

*ML Engineer · Data Engineer · MLOps Engineer · Analytics Engineer*

</div>
