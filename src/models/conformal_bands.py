"""
CASPER-Gov: MAPIE Split Conformal Price Prediction Intervals
============================================================
Wraps the multi-model stacking pipeline in MAPIE's SplitConformalRegressor
to output statistically calibrated low (p10), mid (p50), and high (p90)
price bands guaranteeing 80% coverage intervals.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from mapie.regression import SplitConformalRegressor

from src.models.ensemble_stacking import get_stacking_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("conformal_bands")

FEATURE_COLS = [
    "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
    "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
    "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
    "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
]
CAT_COLS = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]
TARGET_COL = "modal_price_per_quintal"


def prepare_conformal_data(df: pd.DataFrame, cluster_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares and casts feature matrices for conformal pipeline."""
    df_merged = df.copy()
    if cluster_df is not None and "cluster_id" not in df_merged.columns:
        df_merged = pd.merge(df_merged, cluster_df[["sku_name", "cluster_id"]], on="sku_name", how="left")

    if "cluster_id" not in df_merged.columns:
        df_merged["cluster_id"] = "-1"

    for col in FEATURE_COLS:
        if col not in df_merged.columns:
            if col in CAT_COLS:
                df_merged[col] = "Missing"
            else:
                df_merged[col] = 0.0

    X = df_merged[FEATURE_COLS].copy()
    for col in CAT_COLS:
        X[col] = X[col].fillna("Missing").astype(str)

    num_cols = [c for c in FEATURE_COLS if c not in CAT_COLS]
    for col in num_cols:
        X[col] = X[col].astype(float)

    y = df_merged[TARGET_COL].astype(float) if TARGET_COL in df_merged.columns else None
    return X, y


def train_conformal_bands(
    train_path: str = "data/features/train_features.parquet",
    val_path: str = "data/features/val_features.parquet",
    test_path: str = "data/features/test_features.parquet",
    cluster_path: str = "data/features/commodity_clusters.parquet",
    model_save_path: str = "models/mapie_conformal.joblib",
    alpha: float = 0.20, # 80% coverage interval
) -> float:
    """
    Fits the stacking regressor on training set, calibrates on validation set
    using SplitConformalRegressor, serializes the MAPIE model, and computes test coverage.
    """
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    if len(df_train) > 100000:
        logger.info("Downsampling train features from %d to 100000 rows for speed...", len(df_train))
        df_train = df_train.sample(n=100000, random_state=42)
    if len(df_val) > 20000:
        df_val = df_val.sample(n=20000, random_state=42)
    if len(df_test) > 20000:
        df_test = df_test.sample(n=20000, random_state=42)

    cluster_df = pd.read_parquet(cluster_path) if os.path.exists(cluster_path) else None

    X_train, y_train = prepare_conformal_data(df_train, cluster_df)
    X_val, y_val = prepare_conformal_data(df_val, cluster_df)
    X_test, y_test = prepare_conformal_data(df_test, cluster_df)

    logger.info("Initializing stacking base pipeline...")
    base_pipeline = get_stacking_pipeline()

    logger.info("Fitting base stacking regressor pipeline on train set (%d rows)...", len(X_train))
    base_pipeline.fit(X_train, y_train)

    logger.info("Calibrating SplitConformalRegressor on validation set (%d rows, confidence_level=%.2f)...", len(X_val), 1.0 - alpha)
    mapie = SplitConformalRegressor(
        estimator=base_pipeline,
        confidence_level=1.0 - alpha,
        prefit=True,
    )
    mapie.conformalize(X_val, y_val)

    # Evaluate on test set
    y_pred, y_pis = mapie.predict_interval(X_test)
    y_p10 = y_pis[:, 0, 0]
    y_p90 = y_pis[:, 1, 0]

    coverage = float(np.mean((y_test.values >= y_p10) & (y_test.values <= y_p90)) * 100.0)
    logger.info("Conformal 80%% interval empirical test coverage: %.2f%%", coverage)

    joblib.dump(mapie, model_save_path)
    logger.info("Saved calibrated MAPIE model to %s", model_save_path)

    return coverage
