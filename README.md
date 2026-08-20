# CASPER-Gov: Pricing Intelligence & Regulatory Compliance Platform

CASPER-Gov is a modern, AI-powered regulatory intelligence and compliance auditing engine designed to monitor commodity pricing, detect price gouging, flag anti-competitive cartel activity, simulate policy interventions, and generate court-ready legal notices.

---

## 🚀 Key Platform Capabilities

### 1. Data Ingestion & Analytics Pipeline
- **API Wrappers**: Integrates `agmarknet_api` to ingest daily mandi arrivals and modal prices across Indian regions.
- **Macro Scrapers**: Scrapes indicators (CPI, WPI, Freight Transportation indexes) from FRED/World Bank with fallback CSV handlers.
- **Analytics**: Performs Point-in-Time chronological train/validation/test splits (60/20/20) to prevent data leakage in forecasting.

### 2. Feature Engineering & Goods Clustering
- **Feature Pipeline**: Extracts rolling price lags (7d, 14d, 30d, 90d), volatilities, and harvest cycles.
- **Dimensionality Reduction**: Reduces the macro inflation/freight block to orthogonal components using `scikit-learn` PCA.
- **Pricing Archetypes**: Clusters commodities into pricing profiles (e.g. perishable staples, utilities) using UMAP and HDBSCAN.

### 3. Forecasting & Calibrated Price Bands
- **Stacking Regressor Ensemble**: Blends predictions from LightGBM, XGBoost, and Random Forest base estimators using a Ridge meta-learner.
- **Conformal Calibration**: Wraps the meta-regressor in MAPIE conformal predictors, calibrating price bands to guarantee an 80% coverage interval.
- **Zero-Shot Chronos Forecaster**: Leverages `amazon/chronos-t5-small` to forecast prices 4 weeks ahead and compute projected breach risks.

### 4. Anomaly Detection & Cartel Analysis
- **Price Gouging**: Fits an `IsolationForest` to flag individual cost anomalies.
- **Cartel Detection**: Analyzes rolling cross-vendor correlations to flag synchronized pricing spikes (>2.5 std) across independent vendors.

### 5. Explainable RAG & Warning Generator
- **SHAP Tree Explainer**: Attributes the top-5 feature drivers causing pricing spikes.
- **RAG Precedents Store**: Stores and queries precedents in a local persistent ChromaDB collection.
- **Structured LLM notices**: Utilizes `instructor` with Pydantic to draft structured compliance warning notices.

### 6. Policy Simulation & Auditing
- **Decision Science Simulator**: Implements a UCB1 Multi-Armed Bandit model to simulate pricing stabilization interventions (price ceiling, consumer subsidies, and import duty waivers) under stochastic supply shocks.
- **Auditing Logger**: Tracks computations and anomalies in a persistent SQLite database (`data/audit_log.db`) utilizing `structlog` for JSON renderer output.

---

## 📁 Directory Structure

```text
├── data/                         # Local data store
│   ├── raw/                      # Raw parquet arrivals and vendor registries
│   ├── processed/                # Temporal split datasets (train/val/test)
│   ├── features/                 # Engineered features & UMAP clusters
│   ├── chroma/                   # Persistent ChromaDB collections
│   └── simulations/              # UCB bandit and canned demo outputs
├── models/                       # Serialized scikit-learn, MAPIE, & LGBM models
├── src/                          # Application source code
│   ├── agmarknet_api/            # Agmarknet API client wrapper
│   ├── api/                      # FastAPI endpoints (conformal band outputs)
│   ├── audit/                    # SQLite structlog auditing module
│   ├── dashboard/                # Streamlit monitoring and scenario dashboard
│   ├── data/                     # Ingestion and split pipelines
│   ├── features/                 # PCA and rolling features generators
│   ├── llm/                      # Instructor LLM report notice generators
│   ├── models/                   # LGBM, conformal, clustering, & SHAP modules
│   ├── rag/                      # ChromaDB precedent stores
│   └── simulation/               # UCB Multi-Armed Bandit policy models
├── scripts/                      # Crisis canned scenarios script
├── tests/                        # 38-unit defensibility test suite
├── pyproject.toml                # Project configurations
└── README.md                     # Documentation
```

---

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Seed ChromaDB Precedents Collection**:
   ```bash
   python -m src.rag.vector_store
   ```

---

## ⚡ Execution Commands

### 🐋 Run with Docker (Recommended)
Orchestrate and execute the entire platform in isolated containers using Docker Compose:

1. **Build and Boot Services**:
   ```bash
   docker-compose up --build -d
   ```
   *This automatically builds the images and starts the FastAPI backend (port `8000`) and the Streamlit dashboard (port `8501`) with persistent local directory volume mounts.*

2. **Stop Services**:
   ```bash
   docker-compose down
   ```

---

### 🐍 Run Locally (Alternative)

#### 1. Launch FastAPI Backend
Exposes `/api/v1/price-bands`, `/api/v1/monitoring`, and `/api/v1/risk-analysis` (POST) REST endpoints:
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

#### 2. Launch Streamlit Auditing Dashboard
Includes the Live Audit Monitor, the Scenario Planning forecasting simulator, and the Batch Risk CSV Uploader tab:
```bash
streamlit run src/dashboard/app.py --server.port 8501
```

### 3. Run Live Crisis Demo Scenarios
Triggers A (Fuel Spike), B (Pharma Shortage), and C (Onion Wholesaler Cartel) simulations, writing results to `data/simulations/canned_scenarios_output.json`:
```bash
python -m scripts.canned_scenarios
```

### 4. Run Verification QA Test Suite
Executes all 41 test validations:
```bash
python -m pytest tests/
```