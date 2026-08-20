"""
CASPER-Gov: Unsupervised Commodity Archetype Clustering
========================================================
Extracts aggregate behavioral pricing features (volatility, seasonality amplitude,
elasticity proxy, macro correlation), reduces dimensions via UMAP, and clusters
commodities into structural archetypes using HDBSCAN.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:
    import umap.umap_ as umap

try:
    import hdbscan
except ImportError:
    from sklearn.cluster import HDBSCAN as hdbscan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("goods_clustering")


def compute_commodity_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary pricing and volatility statistics per commodity SKU.
    """
    df = df.copy()
    grouped = df.groupby("sku_name")

    records = []
    for sku, group in grouped:
        prices = group["modal_price_per_quintal"].values
        # 1. Volatility
        if len(prices) > 1:
            returns = np.diff(prices) / np.maximum(prices[:-1], 1e-5)
            vol = float(np.std(returns))
        else:
            vol = 0.02

        # 2. Seasonality amplitude
        if "seasonal_index" in group.columns and len(group["seasonal_index"]) > 1:
            season_amp = float(group["seasonal_index"].max() - group["seasonal_index"].min())
        else:
            season_amp = 0.15

        # 3. Price elasticity proxy: correlation between arrivals and price
        if "arrival_quantity_tonnes" in group.columns and len(group) > 2:
            arr = group["arrival_quantity_tonnes"].values
            corr_mat = np.corrcoef(prices, arr)
            elasticity = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else -0.20
        else:
            elasticity = -0.25

        # 4. Macro correlation
        if "macro_pca_1" in group.columns and len(group) > 2:
            macro = group["macro_pca_1"].values
            corr_m = np.corrcoef(prices, macro)
            macro_corr = float(corr_m[0, 1]) if not np.isnan(corr_m[0, 1]) else 0.30
        else:
            macro_corr = 0.35

        records.append({
            "sku_name": sku,
            "volatility": float(vol),
            "seasonality_amplitude": float(season_amp),
            "price_elasticity_proxy": float(elasticity),
            "macro_correlation": float(macro_corr),
        })

    return pd.DataFrame(records)


def run_goods_clustering(
    input_features_path: str = "data/features/train_features.parquet",
    model_dir: str = "models",
    output_parquet_path: str = "data/features/commodity_clusters.parquet",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fits UMAP and HDBSCAN clustering models and writes cluster metadata to Parquet.
    """
    logger.info("Starting goods clustering on %s...", input_features_path)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_parquet_path) or ".", exist_ok=True)

    df_feats = pd.read_parquet(input_features_path)
    df_agg = compute_commodity_aggregates(df_feats)
    
    feature_cols = ["volatility", "seasonality_amplitude", "price_elasticity_proxy", "macro_correlation"]
    X = df_agg[feature_cols].values

    # 1. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. UMAP dimensionality reduction
    n_samples = len(df_agg)
    n_neighbors = min(15, max(2, n_samples - 1))
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=0.1,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X_scaled)

    # 3. HDBSCAN Clustering
    min_cluster_size = max(2, min(3, n_samples // 2))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
    )
    cluster_labels = clusterer.fit_predict(embedding)
    probabilities = getattr(clusterer, "probabilities_", np.ones(len(cluster_labels)))

    # Save models
    umap_payload = {"scaler": scaler, "umap": reducer}
    joblib.dump(umap_payload, os.path.join(model_dir, "umap_goods.joblib"))
    joblib.dump(clusterer, os.path.join(model_dir, "hdbscan_goods.joblib"))
    logger.info("Saved UMAP & HDBSCAN models to %s", model_dir)

    # Build cluster DataFrame
    df_agg["cluster_id"] = cluster_labels.astype(int)
    df_agg["cluster_probability"] = probabilities.astype(float)
    df_agg["umap_1"] = embedding[:, 0].astype(float)
    df_agg["umap_2"] = embedding[:, 1].astype(float)

    df_agg.to_parquet(output_parquet_path, index=False)
    logger.info("Exported %d commodity cluster assignments to %s", len(df_agg), output_parquet_path)

    return df_agg
