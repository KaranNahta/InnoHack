"""
CASPER-Gov: LightGBM Quantile Price Band Regressors
===================================================
Trains LightGBM models optimizing pinball loss for quantiles p10 (floor),
p50 (median fair price), and p90 (statutory ceiling).
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("lightgbm_quantile")

FEATURE_COLS = [
    "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
    "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
    "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
    "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
]
CAT_COLS = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]
TARGET_COL = "modal_price_per_quintal"


def prepare_features(df: pd.DataFrame, cluster_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares and casts feature matrices."""
    df_merged = df.copy()
    if cluster_df is not None and "cluster_id" not in df_merged.columns:
        df_merged = pd.merge(df_merged, cluster_df[["sku_name", "cluster_id"]], on="sku_name", how="left")
    
    if "cluster_id" not in df_merged.columns:
        df_merged["cluster_id"] = -1

    for col in FEATURE_COLS:
        if col not in df_merged.columns:
            if col in CAT_COLS:
                df_merged[col] = "Missing"
            else:
                df_merged[col] = 0.0

    X = df_merged[FEATURE_COLS].copy()
    for col in CAT_COLS:
        X[col] = X[col].fillna("Missing").astype(str).astype("category")

    num_cols = [c for c in FEATURE_COLS if c not in CAT_COLS]
    for col in num_cols:
        X[col] = X[col].astype(float)

    y = df_merged[TARGET_COL].astype(float) if TARGET_COL in df_merged.columns else None
    return X, y


def train_quantile_models(
    train_path: str = "data/features/train_features.parquet",
    val_path: str = "data/features/val_features.parquet",
    test_path: str = "data/features/test_features.parquet",
    cluster_path: str = "data/features/commodity_clusters.parquet",
    model_dir: str = "models",
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Trains LightGBM models for quantiles p10, p50, and p90.
    Serializes models and returns test MAPE and 80% coverage interval percentage.
    """
    os.makedirs(model_dir, exist_ok=True)
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    cluster_df = pd.read_parquet(cluster_path) if os.path.exists(cluster_path) else None

    X_train, y_train = prepare_features(df_train, cluster_df)
    X_val, y_val = prepare_features(df_val, cluster_df)
    X_test, y_test = prepare_features(df_test, cluster_df)

    quantiles = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
    models = {}

    for name, alpha in quantiles.items():
        logger.info("Training LightGBM model for quantile %s (alpha=%.2f)...", name, alpha)
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
        )
        models[name] = model
        save_path = os.path.join(model_dir, f"lgb_{name}.joblib")
        joblib.dump(model, save_path)
        logger.info("Saved %s to %s", name, save_path)

    # Predictions on test set
    y_pred_p10 = models["p10"].predict(X_test)
    y_pred_p50 = models["p50"].predict(X_test)
    y_pred_p90 = models["p90"].predict(X_test)

    # Post-process monotonicity check: ensure p10 <= p50 <= p90
    y_pred_p10_fixed = np.minimum(y_pred_p10, y_pred_p50)
    y_pred_p90_fixed = np.maximum(y_pred_p90, y_pred_p50)

    # Metrics
    mape = float(np.mean(np.abs((y_test.values - y_pred_p50) / np.maximum(y_test.values, 1e-5))) * 100.0)
    coverage = float(np.mean((y_test.values >= y_pred_p10_fixed) & (y_test.values <= y_pred_p90_fixed)) * 100.0)

    logger.info("Quantile modeling complete. Test MAPE: %.2f%%, 80%% Interval Coverage: %.2f%%", mape, coverage)
    return mape, coverage
