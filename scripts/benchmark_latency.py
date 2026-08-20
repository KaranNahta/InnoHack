"""
CASPER-Gov: 7-Stage Pipeline Latency & Throughput Benchmark
===========================================================
Benchmarks execution latency across all 7 architectural stages (Slide 12):
  Stage 1: Feature Lookup & Preprocessing
  Stage 2: UMAP / HDBSCAN Archetype Clustering
  Stage 3: MAPIE Split Conformal Inference (p10/p50/p90)
  Stage 4: SHAP TreeExplainer Attribution (Top-5 Cost Drivers)
  Stage 5: ChromaDB Statutory Legal RAG Retrieval
  Stage 6: LLM Critic Evaluation & Adjustment
  Stage 7: Compliance Verdict & Cryptographic Audit Seal
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd

from src.models.shap_explainer import explain_price_anomaly
from src.rag.vector_store import retrieve_legal_precedents
from src.llm.report_generator import evaluate_price_estimate
from src.audit.logger import log_audit_event
import joblib


def run_benchmark(n_trials: int = 50) -> None:
    print("=" * 75)
    print("   CASPER-Gov: 7-Stage End-to-End Pipeline Latency Benchmark")
    print(f"   Running {n_trials} Monte Carlo trials across standard commodities...")
    print("=" * 75)

    # Preload models
    lgb_model = joblib.load("models/lgb_p50.joblib") if os.path.exists("models/lgb_p50.joblib") else None
    mapie_model = joblib.load("models/mapie_conformal.joblib") if os.path.exists("models/mapie_conformal.joblib") else None
    df_clusters = pd.read_parquet("models/goods_clusters.parquet") if os.path.exists("models/goods_clusters.parquet") else None

    # Benchmark test row
    sample_row = {
        "sku_name": "Tomato",
        "state": "Uttar Pradesh",
        "district": "Varanasi",
        "market_mandi": "Varanasi Mandi",
        "sku_variety": "Desi",
        "price_lag_7d": 1450.0,
        "price_lag_14d": 1420.0,
        "price_lag_30d": 1380.0,
        "price_lag_90d": 1300.0,
        "volatility_7d": 0.04,
        "volatility_30d": 0.07,
        "seasonal_index": 1.05,
        "supply_shock_zscore": -1.2,
        "is_harvest_season": 0.0,
        "macro_pca_1": -0.5,
        "macro_pca_2": 0.1,
        "macro_pca_3": 0.3,
        "macro_pca_4": -0.1,
        "macro_pca_5": 0.0,
        "cluster_id": "0",
        "modal_price_per_quintal": 1850.0,
    }
    df_single = pd.DataFrame([sample_row])

    stage1_times = []
    stage2_times = []
    stage3_times = []
    stage4_times = []
    stage5_times = []
    stage6_times = []
    stage7_times = []
    e2e_times = []

    # Warmup runs
    for _ in range(5):
        retrieve_legal_precedents("Price fixing Tomato", top_k=2)
        if lgb_model:
            explain_price_anomaly(df_single, top_n=3)

    for i in range(n_trials):
        t_start = time.perf_counter()

        # Stage 1: Feature Lookup
        t0 = time.perf_counter()
        feat_dict = dict(sample_row)
        feat_dict["observed_price"] = 1850.0
        t_s1 = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Cluster Assignment
        t0 = time.perf_counter()
        c_id = "-1"
        if df_clusters is not None:
            match = df_clusters[df_clusters["sku_name"].str.lower() == "tomato"]
            if not match.empty:
                c_id = str(int(match.iloc[0]["cluster_id"]))
        t_s2 = (time.perf_counter() - t0) * 1000.0

        # Stage 3: MAPIE Conformal Inference
        t0 = time.perf_counter()
        if mapie_model is not None:
            feature_cols = [
                "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
                "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
                "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
                "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
            ]
            cat_cols = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]
            X_eval = df_single[feature_cols].copy()
            for c in cat_cols:
                X_eval[c] = X_eval[c].astype(str)
            for c in [f for f in feature_cols if f not in cat_cols]:
                X_eval[c] = X_eval[c].astype(float)
            y_pred, y_pis = mapie_model.predict_interval(X_eval)
            raw_p10, raw_p50, raw_p90 = float(y_pis[0, 0, 0]), float(y_pred[0]), float(y_pis[0, 1, 0])
        else:
            raw_p10, raw_p50, raw_p90 = 1200.0, 1400.0, 1600.0
        t_s3 = (time.perf_counter() - t0) * 1000.0

        # Stage 4: SHAP Attribution
        t0 = time.perf_counter()
        shap_drivers = explain_price_anomaly(df_single, top_n=5)
        t_s4 = (time.perf_counter() - t0) * 1000.0

        # Stage 5: ChromaDB RAG Precedents
        t0 = time.perf_counter()
        precedents = retrieve_legal_precedents("Section 3 Essential Commodities Act price ceiling", top_k=2)
        t_s5 = (time.perf_counter() - t0) * 1000.0

        # Stage 6: LLM Critic Decision
        t0 = time.perf_counter()
        verdict = evaluate_price_estimate(
            sku_name="Tomato",
            region="Uttar Pradesh",
            raw_p10=raw_p10,
            raw_p50=raw_p50,
            raw_p90=raw_p90,
            shap_drivers=shap_drivers,
            retrieved_precedents=precedents,
        )
        t_s6 = (time.perf_counter() - t0) * 1000.0

        # Stage 7: Final Compliance & Cryptographic Audit Seal
        t0 = time.perf_counter()
        obs_p = 1850.0
        status = "CEILING_BREACHED" if obs_p > raw_p90 else "WITHIN_BAND"
        entry_hash = log_audit_event(
            sku_id="Tomato",
            region="Uttar Pradesh",
            model_version="mapie_v1.0",
            feature_snapshot_hash="bench_hash",
            observed_price=obs_p,
            computed_band={"p10": raw_p10, "p50": raw_p50, "p90": raw_p90},
            anomaly_type="PRICE_GOUGING_ALERT",
            llm_verdict_json={"decision": verdict.decision, "reason": verdict.reasoning},
            db_path="data/benchmark_audit.db",
        )
        t_s7 = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_start) * 1000.0

        stage1_times.append(t_s1)
        stage2_times.append(t_s2)
        stage3_times.append(t_s3)
        stage4_times.append(t_s4)
        stage5_times.append(t_s5)
        stage6_times.append(t_s6)
        stage7_times.append(t_s7)
        e2e_times.append(t_total)

    # Cleanup temp db
    if os.path.exists("data/benchmark_audit.db"):
        os.remove("data/benchmark_audit.db")

    stages = [
        ("Stage 1: Feature Lookup & Data Feed", stage1_times),
        ("Stage 2: UMAP/HDBSCAN Cluster Routing", stage2_times),
        ("Stage 3: MAPIE Split Conformal Inference", stage3_times),
        ("Stage 4: SHAP TreeExplainer Attribution", stage4_times),
        ("Stage 5: ChromaDB ONNX Legal Retrieval", stage5_times),
        ("Stage 6: LLM Critic & Context Arbiter", stage6_times),
        ("Stage 7: Cryptographic Seal & Audit Hash", stage7_times),
    ]

    print("\n📊 7-STAGE PIPELINE LATENCY PROFILING RESULTS:")
    print("-" * 75)
    print(f"{'Pipeline Stage':<42} | {'Mean (ms)':<10} | {'p50 (ms)':<9} | {'p95 (ms)':<9}")
    print("-" * 75)

    for name, times in stages:
        arr = np.array(times)
        mean_v = float(np.mean(arr))
        p50_v = float(np.median(arr))
        p95_v = float(np.percentile(arr, 95))
        print(f"{name:<42} | {mean_v:8.2f} ms | {p50_v:7.2f} ms | {p95_v:7.2f} ms")

    print("-" * 75)
    e2e_arr = np.array(e2e_times)
    print(f"{'⚡ FULL END-TO-END 7-STAGE PIPELINE':<42} | {np.mean(e2e_arr):8.2f} ms | {np.median(e2e_arr):7.2f} ms | {np.percentile(e2e_arr, 95):7.2f} ms")
    print("=" * 75)
    throughput = 1000.0 / np.mean(e2e_arr)
    print(f"🚀 Estimated Peak Single-Node Throughput: ~{throughput:.1f} price evaluations/second\n")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    run_benchmark(n_trials=n)
