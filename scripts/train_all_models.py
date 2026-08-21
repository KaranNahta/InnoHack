"""
CASPER-Gov: Unified End-to-End Model Training & Artifact Generator
==================================================================
Executes the full pipeline in order:
  1. Ingest/ensure macro datasets & international benchmarks
  2. Ingest/synthesize Agmarknet multi-state mandi price records
  3. Fit & serialize Macro PCA (models/pca_macro.joblib)
  4. Build temporal feature tables (data/features/train_features.parquet, etc.)
  5. Fit UMAP + HDBSCAN commodity archetype clustering (models/goods_clusters.parquet)
  6. Train LightGBM Quantile Regressors p10/p50/p90 (models/lgb_p10.joblib, etc.)
  7. Train Stacking Regressor & MAPIE Split Conformal Predictor (models/mapie_conformal.joblib)
  8. Fit Isolation Forest Anomaly Detector (models/isolation_forest.joblib)
  9. Seed persistent ChromaDB regulatory precedents (data/chromadb/)
"""

from __future__ import annotations

import os
import sys
import time
import logging
import pandas as pd
import numpy as np

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_all_models")


def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("   CASPER-Gov: Unified Model Training & Artifact Generation")
    logger.info("=" * 70)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/raw/macro", exist_ok=True)
    os.makedirs("data/raw/reference", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/features", exist_ok=True)
    os.makedirs("data/chromadb", exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Ensure Raw Macro & Reference Datasets
    # -------------------------------------------------------------------------
    logger.info("\n>>> Step 1/9: Ensuring Macro & Reference Datasets...")
    from src.data.macro_ingest import ensure_fallback_files, run_macro_ingest
    from src.data.reference_data import get_international_benchmarks, get_mrp_ceilings, get_historical_price_controls

    ensure_fallback_files()
    run_macro_ingest()

    # Save international benchmarks & reference files
    df_bench = get_international_benchmarks()
    df_bench.to_parquet("data/raw/reference/international_benchmarks.parquet", index=False)
    get_mrp_ceilings().to_parquet("data/raw/reference/mrp_ceilings.parquet", index=False)
    get_historical_price_controls().to_parquet("data/raw/reference/price_controls.parquet", index=False)
    logger.info("Saved macro and reference datasets.")

    # -------------------------------------------------------------------------
    # Step 2: Ensure Agmarknet Mandi Split Datasets
    # -------------------------------------------------------------------------
    logger.info("\n>>> Step 2/9: Generating/Loading Mandi Dataset & Temporal Splits...")
    train_path = "data/processed/train.parquet"
    val_path = "data/processed/val.parquet"
    test_path = "data/processed/test.parquet"

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        commodities = [
            ("Tomato", 1200.0, 350.0, 45.0),
            ("Potato", 1400.0, 200.0, 80.0),
            ("Onion", 2200.0, 600.0, 60.0),
            ("Wheat", 2100.0, 150.0, 120.0),
            ("Rice", 3100.0, 250.0, 110.0),
            ("Mustard Oil", 13500.0, 800.0, 25.0),
            ("Gram Dal", 6800.0, 450.0, 35.0),
            ("Sugar", 3800.0, 180.0, 75.0),
            ("Turmeric", 7500.0, 500.0, 30.0),
            ("Cotton", 6200.0, 400.0, 50.0),
            ("Maize", 1900.0, 150.0, 95.0),
            ("Soyabean", 4300.0, 300.0, 85.0),
            ("Groundnut", 5800.0, 400.0, 40.0),
            ("Moong Dal", 7800.0, 550.0, 20.0),
            ("Urad Dal", 7200.0, 500.0, 22.0),
            ("Apple", 9500.0, 1200.0, 15.0),
        ]
        states_mandis = [
            ("Uttar Pradesh", "Varanasi Mandi", "Varanasi"),
            ("Uttar Pradesh", "Lucknow Mandi", "Lucknow"),
            ("Punjab", "Ludhiana Mandi", "Ludhiana"),
            ("Haryana", "Karnal Mandi", "Karnal"),
            ("Maharashtra", "Pune Mandi", "Pune"),
            ("Maharashtra", "Nasik Mandi", "Nashik"),
            ("Gujarat", "Ahmedabad Mandi", "Ahmedabad"),
            ("Karnataka", "Bangalore Mandi", "Bangalore Urban"),
            ("Madhya Pradesh", "Indore Mandi", "Indore"),
            ("Rajasthan", "Jaipur Mandi", "Jaipur"),
            ("Tamil Nadu", "Koyambedu Mandi", "Chennai"),
            ("Andhra Pradesh", "Guntur Mandi", "Guntur"),
            ("Bihar", "Patna Mandi", "Patna"),
            ("West Bengal", "Kolkata Mandi", "Kolkata"),
            ("Kerala", "Ernakulam Mandi", "Ernakulam"),
            ("Telangana", "Bowenpally Mandi", "Hyderabad"),
            ("Odisha", "Bhubaneswar Mandi", "Bhubaneswar"),
        ]

        dates = pd.date_range("2024-01-01", "2026-08-15", freq="D")
        records = []
        np.random.seed(42)

        for sku, base_p, p_std, base_arr in commodities:
            for state, mandi, dist in states_mandis:
                reg_factor = 1.0 + (hash(state) % 15 - 7) * 0.01
                p_series = np.clip(
                    np.random.normal(base_p * reg_factor, p_std, len(dates)),
                    base_p * 0.4,
                    base_p * 2.5
                )
                day_of_year = dates.dayofyear.values
                season_wave = np.sin(2 * np.pi * day_of_year / 365.25) * (p_std * 0.6)
                p_series += season_wave

                arr_series = np.clip(
                    np.random.normal(base_arr, base_arr * 0.3, len(dates)),
                    base_arr * 0.1,
                    base_arr * 3.0
                )

                for d, p_val, a_val in zip(dates, p_series, arr_series):
                    records.append({
                        "observation_date": d,
                        "sku_name": sku,
                        "sku_variety": "Standard",
                        "grade": "FAQ",
                        "state": state,
                        "district": dist,
                        "market_mandi": mandi,
                        "arrival_quantity_tonnes": round(float(a_val), 2),
                        "modal_price_per_quintal": round(float(p_val), 2),
                        "min_price_per_quintal": round(float(p_val * 0.93), 2),
                        "max_price_per_quintal": round(float(p_val * 1.07), 2),
                    })

        df_all = pd.DataFrame(records)
        df_all["observation_date"] = pd.to_datetime(df_all["observation_date"])
        df_all = df_all.sort_values("observation_date").reset_index(drop=True)

        n = len(df_all)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        df_train = df_all.iloc[:n_train]
        df_val = df_all.iloc[n_train:n_train + n_val]
        df_test = df_all.iloc[n_train + n_val:]

        df_train.to_parquet(train_path, index=False)
        df_val.to_parquet(val_path, index=False)
        df_test.to_parquet(test_path, index=False)
        logger.info("Created splits: Train=%d, Val=%d, Test=%d", len(df_train), len(df_val), len(df_test))
    else:
        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)
        df_test = pd.read_parquet(test_path)
        logger.info("Loaded existing splits: Train=%d, Val=%d, Test=%d", len(df_train), len(df_val), len(df_test))

    # -------------------------------------------------------------------------
    # Step 3: Fit Macro PCA
    # -------------------------------------------------------------------------
    logger.info("\n>>> Step 3/9: Fitting & Serializing Macro PCA...")
    from src.features.macro_pca import fit_save_macro_pca
    fit_save_macro_pca(
        macro_dir="data/raw/macro",
        ref_dir="data/raw/reference",
        model_path="models/pca_macro.joblib"
    )

    # -------------------------------------------------------------------------
    # Step 4: Build Engineered Features
    # -------------------------------------------------------------------------
    logger.info("\n>>> Step 4/9: Building Lag, Volatility & Macro Features...")
    from src.features.build_features import transform_features
    df_train_feat = transform_features(df_train, "data/raw/macro", "data/raw/reference", "models/pca_macro.joblib")
    df_val_feat = transform_features(df_val, "data/raw/macro", "data/raw/reference", "models/pca_macro.joblib")
    df_test_feat = transform_features(df_test, "data/raw/macro", "data/raw/reference", "models/pca_macro.joblib")

    feat_train_path = "data/features/train_features.parquet"
    feat_val_path = "data/features/val_features.parquet"
    feat_test_path = "data/features/test_features.parquet"

    df_train_feat.to_parquet(feat_train_path, index=False)
    df_val_feat.to_parquet(feat_val_path, index=False)
    df_test_feat.to_parquet(feat_test_path, index=False)
    logger.info("Saved feature tables (Train features=%d, Test features=%d)", len(df_train_feat), len(df_test_feat))

    # -------------------------------------------------------------------------
    # Step 5: Goods Archetype Clustering (UMAP + HDBSCAN)
    # -------------------------------------------------------------------------
    logger.info("\n>>> Step 5/9: Running Unsupervised Archetype Clustering (UMAP + HDBSCAN)...")
    from src.models.goods_clustering import run_goods_clustering
    df_clusters = run_goods_clustering(
        input_features_path=feat_train_path,
        model_dir="models",
        output_parquet_path="data/features/commodity_clusters.parquet",
    )
    logger.info("Clustered %d commodities into archetypes.", len(df_clusters))

    # -------------------------------------------------------------------------
    # Step 6: Train LightGBM Quantile Models (p10, p50, p90)
    # -------------------------------------------------------------------------
    logger.info("\n>>> Step 6/9: Training LightGBM Quantile Regressors (p10, p50, p90)...")
    from src.models.lightgbm_quantile import train_quantile_models
    mape, q_coverage = train_quantile_models(
        train_path=feat_train_path,
        val_path=feat_val_path,
        test_path=feat_test_path,
        cluster_path="data/features/commodity_clusters.parquet",
        model_dir="models",
    )
    logger.info("Quantile models trained (Test MAPE: %.2f%%, Coverage: %.2f%%)", mape, q_coverage)

    # -------------------------------------------------------------------------
    # Step 7: Train Stacking Regressor & MAPIE Conformal Bands
    # -------------------------------------------------------------------------
    logger.info("\n>>> Step 7/9: Training Stacking Regressor & MAPIE Conformal Intervals...")
    from src.models.conformal_bands import train_conformal_bands
    mapie_coverage = train_conformal_bands(
        train_path=feat_train_path,
        val_path=feat_val_path,
        test_path=feat_test_path,
        cluster_path="data/features/commodity_clusters.parquet",
        model_save_path="models/mapie_conformal.joblib",
    )
    logger.info("MAPIE Conformal predictor trained (Test Coverage: %.2f%%).", mapie_coverage)

    # -------------------------------------------------------------------------
    # Step 8: Fit Isolation Forest Anomaly Detector
    # -------------------------------------------------------------------------
    logger.info("\n>>> Step 8/9: Fitting Multi-Signal Isolation Forest Anomaly Detector...")
    from src.models.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector()
    detector.fit(df_train_feat)
    detector.save("models/isolation_forest.joblib")
    logger.info("Saved Isolation Forest model to models/isolation_forest.joblib")

    # -------------------------------------------------------------------------
    # Step 9: Seed Persistent ChromaDB Regulatory Precedents
    # -------------------------------------------------------------------------
    logger.info("\n>>> Step 9/9: Seeding ChromaDB Regulatory Precedents Vector Store...")
    from src.rag.vector_store import populate_database
    populate_database(db_path="data/chromadb")
    logger.info("ChromaDB vector store seeded successfully.")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    elapsed = round(time.time() - t_start, 2)
    logger.info("\n" + "=" * 70)
    logger.info("   ALL CASPER-Gov MODELS & ARTIFACTS GENERATED SUCCESSFULLY!")
    logger.info("   Total execution time: %s seconds", elapsed)
    logger.info("   Generated Artifacts in models/:")
    for f in sorted(os.listdir("models")):
        fpath = os.path.join("models", f)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            logger.info("     • %-30s (%6.1f KB)", f, size_kb)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
