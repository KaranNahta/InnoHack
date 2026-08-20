import os
import json
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

from src.models.anomaly_detector import detect_anomalies

@pytest.fixture
def mock_anomaly_data(tmp_path):
    features_path = os.path.join(tmp_path, "test_features.parquet")
    vendor_path = os.path.join(tmp_path, "vendor_registry.parquet")
    
    # 1. Mock Vendor Registry
    vendors = [
        {"vendor_id": "VEND_A", "licensed_status": "Licensed", "retailer_type": "wholesaler", "region": "Punjab", "registered_skus": "Rice"},
        {"vendor_id": "VEND_B", "licensed_status": "Licensed", "retailer_type": "wholesaler", "region": "Punjab", "registered_skus": "Rice"},
        {"vendor_id": "VEND_C", "licensed_status": "Licensed", "retailer_type": "wholesaler", "region": "Punjab", "registered_skus": "Rice"},
        {"vendor_id": "VEND_D", "licensed_status": "Licensed", "retailer_type": "wholesaler", "region": "Punjab", "registered_skus": "Rice"}
    ]
    pd.DataFrame(vendors).to_parquet(vendor_path, index=False)
    
    # 2. Mock Test Features (Time-series data for 15 days)
    dates = pd.date_range(start="2026-08-01", periods=15, freq="D")
    mandis = ["Mandi A", "Mandi B", "Mandi C", "Mandi D"]
    
    records = []
    for dt in dates:
        # Standard daily pricing
        for idx, mandi in enumerate(mandis):
            price = 3000.0
            
            # Inject Price Gouging anomaly: Mandi A on 2026-08-10 goes up by 80% (individual outlier)
            if dt.strftime("%Y-%m-%d") == "2026-08-10" and mandi == "Mandi A":
                price = 5400.0
                
            # Inject Cartel Spike anomaly: Mandi B, C, D on 2026-08-14 all spike together by 30%
            if dt.strftime("%Y-%m-%d") == "2026-08-14" and mandi in ["Mandi B", "Mandi C", "Mandi D"]:
                price = 3900.0
                
            records.append({
                "observation_date": dt,
                "sku_name": "Rice",
                "state": "Punjab",
                "market_mandi": mandi,
                "district": "Amritsar",
                "sku_variety": "Basmati",
                "modal_price_per_quintal": price,
                "arrival_quantity_tonnes": 100.0,
                "macro_pca_1": 0.0 # Cost driver does not spike
            })
            
    pd.DataFrame(records).to_parquet(features_path, index=False)
    return features_path, vendor_path

def test_anomaly_detection_pipeline(mock_anomaly_data, tmp_path):
    features_path, vendor_path = mock_anomaly_data
    model_save = os.path.join(tmp_path, "isolation_forest.joblib")
    report_save = os.path.join(tmp_path, "price_anomalies.json")
    
    # Run pipeline
    anomalies = detect_anomalies(
        features_path=features_path,
        vendor_path=vendor_path,
        model_save_path=model_save,
        report_save_path=report_save
    )
    
    # Verify outputs
    assert os.path.exists(model_save)
    assert os.path.exists(report_save)
    
    # Verify model loads
    model = joblib.load(model_save)
    assert isinstance(model, IsolationForest)
    
    # Verify report contents
    with open(report_save, "r") as f:
        report_data = json.load(f)
        
    assert isinstance(report_data, list)
    assert len(report_data) > 0
    
    # Check flags raised
    flag_types = [a["anomaly_type"] for a in report_data]
    assert "PRICE_GOUGING_ALERT" in flag_types
    assert "CARTEL_BEHAVIOR_FLAG" in flag_types
    
    # Verify cartel details
    cartel_anoms = [a for a in report_data if a["anomaly_type"] == "CARTEL_BEHAVIOR_FLAG"]
    assert len(cartel_anoms) > 0
    assert "2026-08-14" in [c["date"] for c in cartel_anoms]
    
    # Verify price gouging details
    gouging_anoms = [a for a in report_data if a["anomaly_type"] == "PRICE_GOUGING_ALERT"]
    assert len(gouging_anoms) > 0
    assert "2026-08-10" in [g["date"] for g in gouging_anoms]
