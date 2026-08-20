import os
import pytest
import pandas as pd
import numpy as np
import joblib

FEATURE_COLS = [
    "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
    "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
    "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
    "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
]

CAT_COLS = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]

def test_lgb_quantile_price_band_ordering():
    # 1. Check models
    for name in ["p10", "p50", "p90"]:
        if not os.path.exists(f"models/lgb_{name}.joblib"):
            pytest.skip(f"lgb_{name}.joblib model not trained yet, skipping.")
            
    # Load models
    models = {name: joblib.load(f"models/lgb_{name}.joblib") for name in ["p10", "p50", "p90"]}
    
    # 2. Check test features
    feat_path = "data/features/test_features.parquet"
    if not os.path.exists(feat_path):
        pytest.skip("test_features.parquet not generated yet, skipping.")
        
    df_test = pd.read_parquet(feat_path)
    
    # Slice first 100 rows for quick check
    df_slice = df_test.head(100).copy()
    
    # Preprocess categories
    if os.path.exists("data/features/commodity_clusters.parquet"):
        df_clusters = pd.read_parquet("data/features/commodity_clusters.parquet")
        df_slice = pd.merge(df_slice, df_clusters[["sku_name", "cluster_id"]], on="sku_name", how="left")
    if "cluster_id" not in df_slice.columns:
        df_slice["cluster_id"] = -1
        
    X = df_slice[FEATURE_COLS].copy()
        
    for col in CAT_COLS:
        X[col] = X[col].fillna("Missing").astype(str).astype("category")
        
    # Predict
    y_p10 = models["p10"].predict(X)
    y_p50 = models["p50"].predict(X)
    y_p90 = models["p90"].predict(X)
    
    # 3. Assert price band ordering
    # Quantiles: p10 <= p50 <= p90
    assert np.all(y_p10 <= y_p50), "Found instances where p10 floor was greater than p50 midpoint."
    assert np.all(y_p50 <= y_p90), "Found instances where p50 midpoint was greater than p90 ceiling."
    assert np.all(y_p10 <= y_p90), "Found instances where p10 floor was greater than p90 ceiling."

def test_conformal_mapie_price_band_ordering():
    conformal_path = "models/mapie_conformal.joblib"
    if not os.path.exists(conformal_path):
        pytest.skip("mapie_conformal.joblib model not trained yet, skipping.")
        
    model = joblib.load(conformal_path)
    
    feat_path = "data/features/test_features.parquet"
    if not os.path.exists(feat_path):
        pytest.skip("test_features.parquet not generated yet, skipping.")
        
    df_test = pd.read_parquet(feat_path)
    df_slice = df_test.head(100).copy()
    
    if os.path.exists("data/features/commodity_clusters.parquet"):
        df_clusters = pd.read_parquet("data/features/commodity_clusters.parquet")
        df_slice = pd.merge(df_slice, df_clusters[["sku_name", "cluster_id"]], on="sku_name", how="left")
    if "cluster_id" not in df_slice.columns:
        df_slice["cluster_id"] = -1
        
    X = df_slice[FEATURE_COLS].copy()
        
    # Conformal pipeline preprocessor expects string dtypes for categoricals
    for col in CAT_COLS:
        X[col] = X[col].fillna("Missing").astype(str)
    num_cols = [f for f in FEATURE_COLS if f not in CAT_COLS]
    for col in num_cols:
        X[col] = X[col].astype(float)
        
    # Predict intervals
    y_pred, y_pis = model.predict_interval(X)
    
    y_p10 = y_pis[:, 0, 0]
    y_p50 = y_pred
    y_p90 = y_pis[:, 1, 0]
    
    # Assert price band ordering
    assert np.all(y_p10 <= y_p50), "Found conformal predictions where p10 floor was greater than midpoint."
    assert np.all(y_p50 <= y_p90), "Found conformal predictions where midpoint was greater than p90 ceiling."
