import os
import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch

from src.features.macro_pca import fit_save_macro_pca
from src.features.build_features import transform_features

@pytest.fixture
def mock_macro_data(tmp_path):
    # Create mock CPI, WPI, Freight, and Benchmarks Parquet files
    macro_dir = os.path.join(tmp_path, "macro")
    ref_dir = os.path.join(tmp_path, "reference")
    os.makedirs(macro_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    
    dates = pd.date_range(start="2026-01-01", end="2026-08-01", freq="MS")
    dates_str = [d.strftime("%Y-%m-%d") for d in dates]
    
    pd.DataFrame({
        "observation_date": dates_str,
        "cpi_value": [100.0 + idx for idx in range(len(dates))],
        "series_id": ["CPI_TEST"] * len(dates)
    }).to_parquet(os.path.join(macro_dir, "cpi.parquet"))
    
    pd.DataFrame({
        "observation_date": dates_str,
        "wpi_value": [105.0 + idx for idx in range(len(dates))],
        "series_id": ["WPI_TEST"] * len(dates)
    }).to_parquet(os.path.join(macro_dir, "wpi.parquet"))
    
    pd.DataFrame({
        "observation_date": dates_str,
        "freight_index": [90.0 + (idx * 0.5) for idx in range(len(dates))],
        "index_name": ["FREIGHT_TEST"] * len(dates)
    }).to_parquet(os.path.join(macro_dir, "freight.parquet"))
    
    pd.DataFrame({
        "observation_date": dates_str,
        "fao_food_price_index": [110.0 + idx for idx in range(len(dates))],
        "who_health_indicator": [100.0] * len(dates),
        "iea_energy_index": [95.0 + idx for idx in range(len(dates))]
    }).to_parquet(os.path.join(ref_dir, "international_benchmarks.parquet"))
    
    # Also save harvest calendar
    pd.DataFrame([
        {"sku_name": "Rice", "state": "Punjab", "harvest_start_month": 10, "harvest_end_month": 12, "harvest_season": "Kharif"}
    ]).to_parquet(os.path.join(ref_dir, "harvest.parquet"))
    
    return macro_dir, ref_dir

def test_macro_pca_fitting(mock_macro_data, tmp_path):
    macro_dir, ref_dir = mock_macro_data
    model_path = os.path.join(tmp_path, "pca_macro.joblib")
    
    # Run PCA fit
    fit_save_macro_pca(macro_dir, ref_dir, model_path)
    
    assert os.path.exists(model_path)
    model_data = joblib.load(model_path)
    assert "scaler" in model_data
    assert "pca" in model_data
    assert "feature_names" in model_data
    assert model_data["feature_names"] == ["cpi_value", "wpi_value", "freight_index", "fao_food_price_index", "who_health_indicator", "iea_energy_index"]

def test_transform_features(mock_macro_data, tmp_path):
    macro_dir, ref_dir = mock_macro_data
    model_path = os.path.join(tmp_path, "pca_macro.joblib")
    
    # Run PCA fit first
    fit_save_macro_pca(macro_dir, ref_dir, model_path)
    
    # Generate mock daily observations spanning Jan 2026 to Mar 2026
    dates = pd.date_range(start="2026-01-01", end="2026-03-31", freq="D")
    raw_records = []
    
    for dt in dates:
        raw_records.append({
            "observation_date": dt.strftime("%Y-%m-%d"),
            "sku_name": "Rice",
            "state": "Punjab",
            "market_mandi": "Amritsar Mandi",
            "modal_price_per_quintal": float(3000.0 + np.sin(dt.day) * 50.0),
            "arrival_quantity_tonnes": float(100.0 + np.cos(dt.day) * 10.0)
        })
        
    df_raw = pd.DataFrame(raw_records)
    
    # Transform features
    df_feat = transform_features(df_raw, macro_dir, ref_dir, model_path)
    
    # Check that lag columns are present
    assert "price_lag_7d" in df_feat.columns
    assert "price_lag_14d" in df_feat.columns
    assert "price_lag_30d" in df_feat.columns
    assert "price_lag_90d" in df_feat.columns
    
    # Check that volatility columns are present
    assert "volatility_7d" in df_feat.columns
    assert "volatility_30d" in df_feat.columns
    
    # Check other features
    assert "seasonal_index" in df_feat.columns
    assert "supply_shock_zscore" in df_feat.columns
    assert "is_harvest_season" in df_feat.columns
    
    # Check PCA columns (should be 5 columns since PCA was fit with 5 components)
    for i in range(1, 6):
        assert f"macro_pca_{i}" in df_feat.columns
        assert not df_feat[f"macro_pca_{i}"].isna().any()
