# CASPER-Gov: System Architecture

> AI-Powered Commodity Price Surveillance & Enforcement Platform

---

## 7-Stage End-to-End Pipeline

```mermaid
flowchart TD
    A([🌐 REST Request\nsku / state / observed_price]) --> S1

    subgraph S1["Stage 1 · Feature Lookup & Enrichment  ~0.01 ms"]
        F1[Rolling Lags: 7d 14d 30d 90d]
        F2[Volatility Windows + Harvest Calendar]
        F3[Supply Shock Z-Score]
        F4[Macro PCA Block: CPI · WPI · Freight → 5 components]
        F1 --- F2 --- F3 --- F4
    end

    S1 --> S2

    subgraph S2["Stage 2 · Commodity Archetype Routing  ~0.01 ms"]
        G1[UMAP Dimensionality Reduction]
        G2[HDBSCAN Cluster Assignment]
        G3[Cluster ID injected into feature vector]
        G1 --> G2 --> G3
    end

    S2 --> S3

    subgraph S3["Stage 3 · Conformal Price Band Inference  ~18 ms"]
        H1["Stacking Ensemble\nLightGBM + XGBoost + Random Forest → Ridge Meta-Learner"]
        H2["MAPIE Split Conformal Predictor\n83.65% empirical coverage"]
        H3[p10 floor · p50 median · p90 ceiling]
        H1 --> H2 --> H3
    end

    S3 --> S4

    subgraph S4["Stage 4 · SHAP Attribution  ~28 ms"]
        I1[SHAP TreeExplainer]
        I2[Top-5 Cost Driver Features]
        I3["Feature importance: supply_shock_zscore · fuel_cpi · lag_30d · ..."]
        I1 --> I2 --> I3
    end

    S4 --> S5

    subgraph S5["Stage 5 · Statutory RAG Retrieval  ~57 ms"]
        J1[ChromaDB Local Vector Store]
        J2[ONNX MiniLM Embeddings — no network call]
        J3["ECA 1955 §3 · Competition Act 2002 §3(3)(a) · Legal Metrology Rules 2011"]
        J1 --- J2 --> J3
    end

    S5 --> S6

    subgraph S6["Stage 6 · LLM Critic & Context Arbiter  ~0.02 ms"]
        K1[Pydantic-structured enforcement decision]
        K2[Band adjustments from precedent context]
        K3[Show-cause directive text generation]
        K1 --> K2 --> K3
    end

    S6 --> S7

    subgraph S7["Stage 7 · Cryptographic Audit Seal  ~1.3 ms"]
        L1["SHA-256 Block Hash\nhash_n = SHA256(hash_{n-1} || timestamp || payload)"]
        L2[SQLite tamper-evident ledger]
        L3[Chain verified via GET /api/v1/audit/verify]
        L1 --> L2 --> L3
    end

    S7 --> OUT

    OUT([📋 Response\np10 · p50 · p90 · compliance_status\nshap_drivers · llm_notice · audit_hash])

    style S1 fill:#1e3a5f,color:#e2e8f0
    style S2 fill:#1e3a5f,color:#e2e8f0
    style S3 fill:#2d4a22,color:#e2e8f0
    style S4 fill:#2d4a22,color:#e2e8f0
    style S5 fill:#4a2d22,color:#e2e8f0
    style S6 fill:#4a2d22,color:#e2e8f0
    style S7 fill:#2d2244,color:#e2e8f0
```

---

## Platform Component Map

```mermaid
graph LR
    subgraph Ingestion
        A1[Agmarknet API Client]
        A2[Macro Scrapers: FRED · World Bank]
        A3[Point-in-Time Temporal Splits]
    end

    subgraph ML["ML Core"]
        B1[Macro PCA]
        B2[UMAP + HDBSCAN Clustering]
        B3[LightGBM Quantile p10/p50/p90]
        B4[MAPIE Conformal Calibration]
        B5[Isolation Forest Anomaly Detector]
        B6[Chronos Zero-Shot Forecaster]
    end

    subgraph Intelligence["Legal Intelligence"]
        C1[ChromaDB ONNX Vector Store]
        C2[SHAP TreeExplainer]
        C3[Instructor + LLM Critic]
    end

    subgraph Enforcement["Enforcement Engine"]
        D1[SHA-256 Chained Audit Logger]
        D2[ReportLab PDF Notice Generator]
        D3[Cartel Collusion Network Graph]
        D4[UCB1 Policy Bandit Simulator]
    end

    subgraph API["FastAPI REST Layer"]
        E1["POST /api/v1/price-estimate"]
        E2["GET  /api/v1/price-bands"]
        E3["GET  /api/v1/monitoring"]
        E4["GET  /api/v1/anomalies"]
        E5["POST /api/v1/enforce"]
        E6["POST /api/v1/enforce/pdf"]
        E7["GET  /api/v1/audit/logs"]
        E8["GET  /api/v1/audit/verify"]
        E9["GET  /api/v1/stats"]
        E10["GET  /health"]
    end

    subgraph Dashboard["Streamlit Command Center"]
        F1[Mandi+ Live Monitoring Tab]
        F2[Forecasting & Scenario Planning Tab]
        F3[Batch Risk Upload Tab]
        F4[Cartel Network Analysis Tab]
        F5[Audit Chain Viewer Tab]
        F6[PDF Notice Download]
    end

    Ingestion --> ML
    ML --> Intelligence
    Intelligence --> Enforcement
    Enforcement --> API
    API --> Dashboard
```

---

## Latency Profile (30-trial Monte Carlo Benchmark)

| Stage | Mean | p50 | p95 |
|-------|------|-----|-----|
| Stage 1: Feature Lookup | 0.01 ms | 0.00 ms | 0.00 ms |
| Stage 2: Cluster Routing | 0.01 ms | 0.00 ms | 0.00 ms |
| Stage 3: Conformal Inference | 17.98 ms | 16.97 ms | 17.73 ms |
| Stage 4: SHAP Attribution | 28.40 ms | 28.33 ms | 28.98 ms |
| Stage 5: Legal RAG Retrieval | 56.95 ms | 56.47 ms | 63.98 ms |
| Stage 6: LLM Critic | 0.02 ms | 0.02 ms | 0.04 ms |
| Stage 7: Cryptographic Seal | 1.32 ms | 1.00 ms | 2.70 ms |
| **⚡ Full E2E** | **104.67 ms** | **102.89 ms** | **115.35 ms** |

> Estimated throughput: **~10 evaluations/second** on a single CPU node (no GPU).

---

## Data Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI API
    participant ML as ML Engine
    participant RAG as ChromaDB RAG
    participant LLM as LLM Critic
    participant Audit as Audit Ledger

    Client->>API: POST /api/v1/price-estimate {sku, state, observed_price}
    API->>ML: Feature engineering + UMAP cluster lookup
    ML->>ML: MAPIE conformal inference → p10/p50/p90
    ML->>ML: SHAP TreeExplainer → top-5 drivers
    ML->>RAG: Query statutory precedents (ONNX MiniLM)
    RAG-->>ML: ECA 1955 §3, Competition Act §3(3)(a)
    ML->>LLM: Evaluate price + precedents → enforcement decision
    LLM-->>API: Structured verdict + notice text
    API->>Audit: SHA-256 seal → append block to ledger
    Audit-->>API: entry_hash
    API-->>Client: {p10, p50, p90, compliance_status, shap_drivers, llm_notice, audit_hash}
```

---

## Statutory Legal Basis

| Statute | Section | Enforcement Use |
|---------|---------|----------------|
| Essential Commodities Act, 1955 | §3 | Price ceiling control orders |
| Competition Act, 2002 | §3(3)(a) | Anti-competitive cartel agreements |
| Legal Metrology (Packaged Commodities) Rules, 2011 | Rule 6 | Price declaration compliance |
| Consumer Protection Act, 2019 | §2(9) | Unfair trade practices |
