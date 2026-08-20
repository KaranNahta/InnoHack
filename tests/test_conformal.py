import os
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from mapie.regression import SplitConformalRegressor

from src.models.ensemble_stacking import get_stacking_pipeline
from src.models.conformal_bands import train_conformal_bands

@pytest.fixture
def mock_conformal_data(tmp_path):
    train_path = os.path.join(tmp_path, "train_features.parquet")
    val_path = os.path.join(tmp_path, "val_features.parquet")
    test_path = os.path.join(tmp_path, "test_features.parquet")
    cluster_path = os.path.join(tmp_path, "commodity_clusters.parquet")
    
    # Generate mock features matching the pipeline expectations
    cols = [
        "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
        "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
        "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
        "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id", "modal_price_per_quintal"
    ]
    
    # Create 35 records for training, 15 for validation, 15 for testing
    np.random.seed(42)
    def make_df(n):
        data = {}
        for col in cols[:14]:
            data[col] = np.random.randn(n)
        
        # Categoricals (ensure they are strings)
        data["sku_name"] = np.random.choice(["Rice", "Wheat"], n)
        data["state"] = np.random.choice(["Punjab", "Uttar Pradesh"], n)
        data["district"] = np.random.choice(["District X"], n)
        data["market_mandi"] = np.random.choice(["Mandi X"], n)
        data["sku_variety"] = np.random.choice(["Variety X"], n)
        
        # Target
        data["modal_price_per_quintal"] = np.random.uniform(1500.0, 4500.0, n)
        return pd.DataFrame(data)
        
    make_df(35).to_parquet(train_path, index=False)
    make_df(15).to_parquet(val_path, index=False)
    make_df(15).to_parquet(test_path, index=False)
    
    # Mock clusters
    pd.DataFrame([
        {"sku_name": "Rice", "cluster_id": 0, "cluster_probability": 1.0},
        {"sku_name": "Wheat", "cluster_id": 0, "cluster_probability": 1.0}
    ]).to_parquet(cluster_path, index=False)
    
    return train_path, val_path, test_path, cluster_path

def test_get_stacking_pipeline():
    pipeline = get_stacking_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "stacking" in pipeline.named_steps

def test_train_conformal_bands(mock_conformal_data, tmp_path):
    train_path, val_path, test_path, cluster_path = mock_conformal_data
    model_save_path = os.path.join(tmp_path, "mapie_conformal.joblib")
    
    # Run calibration
    coverage = train_conformal_bands(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        cluster_path=cluster_path,
        model_save_path=model_save_path
    )
    
    # Verify outputs
    assert isinstance(coverage, float)
    assert 0.0 <= coverage <= 100.0
    assert os.path.exists(model_save_path)
    
    # Load and test prediction bounds
    mapie = joblib.load(model_save_path)
    assert isinstance(mapie, SplitConformalRegressor)
    
    # Create single row input df
    X_single = pd.DataFrame([{
        "price_lag_7d": 3000.0, "price_lag_14d": 3000.0, "price_lag_30d": 3000.0, "price_lag_90d": 3000.0,
        "volatility_7d": 0.0, "volatility_30d": 0.0, "seasonal_index": 1.0, "supply_shock_zscore": 0.0, "is_harvest_season": 0.0,
        "macro_pca_1": 0.0, "macro_pca_2": 0.0, "macro_pca_3": 0.0, "macro_pca_4": 0.0, "macro_pca_5": 0.0,
        "sku_name": "Rice", "state": "Punjab", "district": "District X", "market_mandi": "Mandi X", "sku_variety": "Variety X", "cluster_id": "0"
    }])
    
    y_pred, y_pis = mapie.predict_interval(X_single)
    p10 = y_pis[0, 0, 0]
    p50 = y_pred[0]
    p90 = y_pis[0, 1, 0]
    
    assert p10 <= p50 <= p90
