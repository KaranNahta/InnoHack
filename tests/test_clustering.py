import os
import pytest
import pandas as pd
import numpy as np
import joblib

from src.models.goods_clustering import run_goods_clustering

@pytest.fixture
def mock_features_data(tmp_path):
    # Create mock training features with multiple commodities
    feat_path = os.path.join(tmp_path, "train_features.parquet")
    
    commodities = ["Rice", "Wheat", "Potato", "Onion", "Tomato", "Pulses"]
    records = []
    
    # Generate 30 daily observations per commodity
    dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
    
    for comm in commodities:
        # Add some differences between commodities to make clustering meaningful
        base_price = {"Rice": 3000.0, "Wheat": 2200.0, "Potato": 1200.0, "Onion": 1800.0, "Tomato": 1500.0, "Pulses": 6500.0}[comm]
        base_arr = {"Rice": 100.0, "Wheat": 150.0, "Potato": 250.0, "Onion": 200.0, "Tomato": 80.0, "Pulses": 50.0}[comm]
        
        for dt in dates:
            records.append({
                "observation_date": dt.strftime("%Y-%m-%d"),
                "sku_name": comm,
                "state": "Punjab",
                "market_mandi": "Amritsar Mandi",
                "modal_price_per_quintal": float(base_price + np.sin(dt.day) * 50.0),
                "arrival_quantity_tonnes": float(base_arr + np.cos(dt.day) * 10.0),
                "seasonal_index": float(1.0 + np.sin(dt.month) * 0.1),
                "macro_pca_1": float(np.sin(dt.day) * 0.2)
            })
            
    df = pd.DataFrame(records)
    df.to_parquet(feat_path, index=False)
    return feat_path

def test_goods_clustering_pipeline(mock_features_data, tmp_path):
    feat_path = mock_features_data
    model_dir = os.path.join(tmp_path, "models")
    output_parquet = os.path.join(tmp_path, "commodity_clusters.parquet")
    
    # Run clustering
    run_goods_clustering(
        input_features_path=feat_path,
        model_dir=model_dir,
        output_parquet_path=output_parquet
    )
    
    # Verify models exist
    assert os.path.exists(os.path.join(model_dir, "umap_goods.joblib"))
    assert os.path.exists(os.path.join(model_dir, "hdbscan_goods.joblib"))
    
    # Verify joblib model data can be loaded
    umap_data = joblib.load(os.path.join(model_dir, "umap_goods.joblib"))
    assert "scaler" in umap_data
    assert "umap" in umap_data
    
    hdbscan_model = joblib.load(os.path.join(model_dir, "hdbscan_goods.joblib"))
    assert hasattr(hdbscan_model, "labels_")
    
    # Verify output parquet
    assert os.path.exists(output_parquet)
    df_clusters = pd.read_parquet(output_parquet)
    
    # Check shape
    assert len(df_clusters) == 6  # 6 unique commodities
    assert "sku_name" in df_clusters.columns
    assert "cluster_id" in df_clusters.columns
    assert "cluster_probability" in df_clusters.columns
    assert "umap_1" in df_clusters.columns
    assert "umap_2" in df_clusters.columns
    
    # Check that aggregate feature columns exist
    for col in ["volatility", "seasonality_amplitude", "price_elasticity_proxy", "macro_correlation"]:
        assert col in df_clusters.columns
        assert not df_clusters[col].isna().any()
