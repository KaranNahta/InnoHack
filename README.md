# ⚖️ CASPER-Gov: AI-Powered Commodity Price Surveillance & Enforcement Platform

![CI](https://github.com/KaranNahta/InnoHack/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-100%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> **InnoHack 2026 Submission** — Intelligent government-grade regulatory intelligence engine for real-time essential commodity price monitoring, anomaly detection, anti-cartel enforcement, and court-ready legal notice generation.
>
> 📐 [View System Architecture →](ARCHITECTURE.md)

---

## 🚀 One-Click Demo Launch

```bash
git clone https://github.com/KaranNahta/InnoHack.git
cd InnoHack
pip install -r requirements.txt
./run_demo.sh
```

This single command:
1. Verifies all ML model artifacts — trains them automatically if missing
2. Launches **FastAPI ML & Enforcement Engine** → `http://localhost:8000/docs`
3. Launches **Streamlit Interactive Command Center** → `http://localhost:8501`

---

## 🧠 Architecture: 7-Stage Pipeline (End-to-End < 105 ms)

```
Stage 1 → Feature Lookup & Temporal Enrichment          ~0.01 ms
Stage 2 → UMAP/HDBSCAN Commodity Archetype Routing      ~0.01 ms
Stage 3 → MAPIE Split Conformal Inference (p10/p50/p90) ~18 ms
Stage 4 → SHAP TreeExplainer Attribution (Top-5)        ~28 ms
Stage 5 → ChromaDB ONNX Statutory Precedent Retrieval   ~57 ms
Stage 6 → LLM Critic Evaluation & Context Arbitration   ~0.02 ms
Stage 7 → Cryptographic SHA-256 Seal & Audit Block      ~1.3 ms
                                                        ─────────
                              Full E2E (p50):           ~103 ms
                              Throughput:               ~10 req/s
```

---

## 🏛️ Platform Capabilities

### 1. 📊 Data Ingestion & Analytics Pipeline
- **Agmarknet API wrappers** — ingests daily mandi arrivals and modal prices across Indian regions
- **Macro scrapers** — CPI, WPI, and Freight Transportation Indexes from FRED/World Bank with CSV fallbacks
- **Point-in-Time splits** — 60/20/20 train/val/test without data leakage

### 2. ⚙️ Feature Engineering & Goods Clustering
- Rolling price lags: **7d, 14d, 30d, 90d** | Volatility windows | Harvest calendars | Supply shock z-scores
- **Macro PCA** — reduces CPI/WPI/freight block to 5 orthogonal components
- **UMAP + HDBSCAN** — clusters commodities into pricing archetypes (perishable staples, durables, etc.)

### 3. 📈 Calibrated Conformal Price Bands
- **Stacking Ensemble**: LightGBM + XGBoost + Random Forest → Ridge meta-learner
- **MAPIE Split Conformal Predictor** — calibrated p10/p50/p90 bands with **83.65% empirical test coverage**
- **Chronos Zero-Shot Forecaster** (`amazon/chronos-t5-small`) — 4-week price trajectories & projected breach risk

### 4. 🚨 Anomaly Detection & Anti-Cartel Engine
- **Multi-Signal Isolation Forest** — flags individual SKU price gouging events
- **Inter-Mandi Cartel Collusion Network** — Plotly graph of synchronized pricing spikes (r > 0.75) with Competition Act 2002 §3 alerts
- Rolling cross-vendor correlation matrices during supply shock periods

### 5. 🔍 Explainable RAG Legal Intelligence
- **SHAP TreeExplainer** — top-5 cost driver attribution per enforcement decision
- **ChromaDB Local Precedent Store** — ONNX MiniLM retrieval of ECA 1955, Competition Act 2002, Legal Metrology Rules 2011
- **Instructor + LLM** — structured Pydantic compliance notice generation

### 6. 🔐 Cryptographic Tamper-Evident Audit Trail
- **SHA-256 block-chaining**: `hash_n = SHA256(hash_{n-1} || timestamp || sku || region || price || band || verdict)`
- `GET /api/v1/audit/verify` — mathematically verifies unbroken chain from genesis block
- Court-admissible cryptographic provenance for every enforcement decision

### 7. 📄 Court-Ready PDF Enforcement Notice Generator
- Ministry of Consumer Affairs official header format
- Price deviation vs p90 statutory ceiling, SHAP cost driver table, legal citations
- Digital SHA-256 authentication seal + show-cause directive with 48-hour mandate
- `POST /api/v1/enforce/pdf` streaming endpoint + 1-click download in dashboard

### 8. 🎯 Policy Simulation Engine
- **UCB1 Multi-Armed Bandit** — simulates price ceiling, consumer subsidies, and import duty waivers under stochastic supply shocks
- Writes results to `data/simulations/`

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/price-estimate` | **Full 7-stage pipeline** — conformal bands + SHAP + RAG + LLM critic + audit seal |
| `GET`  | `/api/v1/price-bands` | Calibrated p10/p50/p90 conformal price bands |
| `GET`  | `/api/v1/monitoring` | Live price monitoring with anomaly scores |
| `POST` | `/api/v1/enforce` | Generate structured LLM enforcement notice |
| `POST` | `/api/v1/enforce/pdf` | Stream court-ready PDF enforcement order |
| `GET`  | `/api/v1/anomalies` | Isolation Forest anomaly events |
| `POST` | `/api/v1/risk-analysis` | Batch CSV risk upload & scoring |
| `GET`  | `/api/v1/audit/logs` | Cryptographic audit block log viewer |
| `GET`  | `/api/v1/audit/verify` | Mathematical chain integrity verification |

Interactive Swagger docs: `http://localhost:8000/docs`

---

## 📁 Directory Structure

```
├── run_demo.sh                   # ⚡ 1-click demo launcher
├── scripts/
│   ├── train_all_models.py       # 1-command end-to-end model training bootstrapper
│   └── benchmark_latency.py     # 7-stage pipeline SLA latency profiler
├── src/
│   ├── api/main.py               # FastAPI — all 9 REST endpoints
│   ├── audit/logger.py           # SHA-256 chained cryptographic audit trail
│   ├── dashboard/
│   │   ├── app.py                # Streamlit command center (6 tabs)
│   │   └── components/
│   │       └── cartel_graph.py   # Interactive Plotly cartel network visualizer
│   ├── utils/
│   │   └── pdf_exporter.py       # ReportLab court-ready PDF notice generator
│   ├── models/                   # LGBM, MAPIE conformal, clustering, SHAP, Chronos
│   ├── features/                 # Macro PCA + rolling feature engineering
│   ├── rag/                      # ChromaDB ONNX precedent vector store
│   ├── llm/                      # Instructor + LLM structured notice generator
│   ├── data/                     # Ingestion, temporal splits, macro scrapers
│   └── simulation/               # UCB1 Multi-Armed Bandit policy simulator
├── models/                       # Serialized model weights (gitignored)
├── data/                         # Local data store (gitignored)
├── tests/                        # ✅ 100 unit tests — 100% pass rate
└── requirements.txt
```

---

## 🛠️ Installation & Setup

### Option A: Local (Recommended for Demo)

#### For Linux/macOS:
```bash
# 1. Clone and install
git clone https://github.com/KaranNahta/InnoHack.git
cd InnoHack
pip install -r requirements.txt

# 2. One-command model training (builds all artifacts in ~30s)
python scripts/train_all_models.py

# 3. Launch everything
./run_demo.sh
```

#### For Windows (PowerShell):
```powershell
# 1. Clone and install
git clone https://github.com/KaranNahta/InnoHack.git
cd InnoHack
pip install -r requirements.txt

# 2. One-command model training (builds all artifacts in ~30s)
python scripts/train_all_models.py

# 3. Launch FastAPI backend
.venv\Scripts\uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 4. Launch Streamlit dashboard (run in a separate terminal)
.venv\Scripts\streamlit run src/dashboard/app.py --server.port 8501 --server.headless true
```

### Option B: Docker
```bash
docker-compose up --build -d
# FastAPI → http://localhost:8000/docs
# Streamlit → http://localhost:8501
docker-compose down
```

---

## ⚡ CLI Commands

### For Linux/macOS:
```bash
# Train all ML models from scratch
python scripts/train_all_models.py

# Run 7-stage pipeline latency benchmark (30 trials)
PYTHONPATH=. python scripts/benchmark_latency.py 30

# Run full 100-test verification suite
PYTHONPATH=. pytest -v

# Launch API only
PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Launch dashboard only
PYTHONPATH=. streamlit run src/dashboard/app.py
```

### For Windows (PowerShell):
```powershell
# Train all ML models from scratch
python scripts/train_all_models.py

# Run 7-stage pipeline latency benchmark (30 trials)
$env:PYTHONPATH="."
python scripts/benchmark_latency.py 30

# Run full 100-test verification suite
python -m pytest -v

# Launch API only
.venv\Scripts\uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Launch dashboard only
.venv\Scripts\streamlit run src/dashboard/app.py --server.port 8501 --server.headless true
```

---

## ✅ Verification

```
100 tests passed · 0 failed · 13.6s
```

| Component | Tests |
|-----------|-------|
| Data ingestion & validation | 8 |
| Macro PCA & feature engineering | 4 |
| Goods clustering (UMAP + HDBSCAN) | 2 |
| Conformal price bands (MAPIE) | 8 |
| Quantile LightGBM models | 3 |
| Anomaly detection (Isolation Forest) | 22 |
| Chronos price forecaster | 3 |
| RAG vector store & legal schemas | 3 |
| SHAP attribution & LLM critic | 2 |
| FastAPI endpoints & dashboard | 16 |
| Cryptographic audit trail | 3 |
| Court-ready PDF generator | 2 |
| Cartel network visualizer | 1 |
| Batch risk upload | 8 |
| Macro ingest pipeline | 5 |
| Price bands ordering (monotonic) | 4 |
| Upload risk analysis | 6 |

---

## 📜 Legal & Statutory Basis

The platform enforces against the following statutes embedded in the ChromaDB precedent store:
- **Essential Commodities Act, 1955 §3** — price ceiling control orders
- **Competition Act, 2002 §3(3)(a)** — anti-competitive cartel agreements
- **Legal Metrology (Packaged Commodities) Rules, 2011** — price declaration compliance
- **Consumer Protection Act, 2019 §2(9)** — unfair trade practices

---

*Built for InnoHack 2026 · CASPER-Gov: Commodity AI Surveillance & Price Enforcement Regulatory engine*