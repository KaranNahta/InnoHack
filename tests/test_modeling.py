import os
import pytest
import pandas as pd
import numpy as np
import joblib

from src.models.lightgbm_quantile import train_quantile_models

@pytest.fixture
def mock_modeling_data(tmp_path):
    train_path = os.path.join(tmp_path, "train_features.parquet")
    val_path = os.path.join(tmp_path, "val_features.parquet")
    test_path = os.path.join(tmp_path, "test_features.parquet")
    cluster_path = os.path.join(tmp_path, "commodity_clusters.parquet")
    
    # Generate mock features
    cols = [
        "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
        "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
        "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
        "sku_name", "state", "district", "market_mandi", "sku_variety", "modal_price_per_quintal"
    ]
    
    # Create 50 records for training, 20 for validation, 20 for testing
    np.random.seed(42)
    def make_df(n):
        data = {}
        for col in cols[:14]:
            data[col] = np.random.randn(n)
        
        # Categoricals
        data["sku_name"] = np.random.choice(["Rice", "Wheat", "Potato"], n)
        data["state"] = np.random.choice(["Punjab", "Uttar Pradesh"], n)
        data["district"] = np.random.choice(["District A", "District B"], n)
        data["market_mandi"] = np.random.choice(["Mandi A", "Mandi B"], n)
        data["sku_variety"] = np.random.choice(["Variety X", "Variety Y"], n)
        
        # Target (must be positive)
        data["modal_price_per_quintal"] = np.random.uniform(1000.0, 5000.0, n)
        return pd.DataFrame(data)
        
    make_df(50).to_parquet(train_path, index=False)
    make_df(20).to_parquet(val_path, index=False)
    make_df(20).to_parquet(test_path, index=False)
    
    # Mock clusters
    pd.DataFrame([
        {"sku_name": "Rice", "cluster_id": 0, "cluster_probability": 1.0},
        {"sku_name": "Wheat", "cluster_id": 0, "cluster_probability": 1.0},
        {"sku_name": "Potato", "cluster_id": 1, "cluster_probability": 1.0}
    ]).to_parquet(cluster_path, index=False)
    
    return train_path, val_path, test_path, cluster_path

def test_train_quantile_models(mock_modeling_data, tmp_path):
    train_path, val_path, test_path, cluster_path = mock_modeling_data
    model_dir = os.path.join(tmp_path, "models")
    
    # Train
    mape, coverage = train_quantile_models(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        cluster_path=cluster_path,
        model_dir=model_dir
    )
    
    # Verify outputs
    assert isinstance(mape, float)
    assert isinstance(coverage, float)
    assert mape >= 0.0
    assert 0.0 <= coverage <= 100.0
    
    # Verify models exist
    for name in ["p10", "p50", "p90"]:
        p = os.path.join(model_dir, f"lgb_{name}.joblib")
        assert os.path.exists(p)
        model = joblib.load(p)
        assert hasattr(model, "predict")
