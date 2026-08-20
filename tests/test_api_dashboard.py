import os
import pytest
from fastapi.testclient import TestClient

import pandas as pd
import numpy as np
from src.api.main import app, load_resources
import src.api.main as api_module

class DummyModel:
    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
    def predict(self, X):
        return np.full(len(X), 3000.0 * self.multiplier)

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment(tmp_path_factory):
    # Ensure test features directory and parquet file exist
    os.makedirs("data/features", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    # Mock test_features.parquet if missing
    if not os.path.exists(api_module.TEST_FEAT_PATH):
        dates = pd.date_range("2026-08-01", periods=10, freq="D")
        records = []
        for dt in dates:
            for mandi in ["Amritsar Mandi", "Ludhiana Mandi"]:
                records.append({
                    "observation_date": dt,
                    "sku_name": "Rice",
                    "state": "Punjab",
                    "district": "Amritsar",
                    "market_mandi": mandi,
                    "sku_variety": "Basmati",
                    "modal_price_per_quintal": 3200.0,
                    "arrival_quantity_tonnes": 150.0,
                    "price_lag_7d": 3150.0,
                    "price_lag_14d": 3100.0,
                    "price_lag_30d": 3050.0,
                    "price_lag_90d": 3000.0,
                    "volatility_7d": 0.02,
                    "volatility_30d": 0.03,
                    "seasonal_index": 1.0,
                    "supply_shock_zscore": 0.0,
                    "is_harvest_season": 0.0,
                    "macro_pca_1": 0.0,
                    "macro_pca_2": 0.0,
                    "macro_pca_3": 0.0,
                    "macro_pca_4": 0.0,
                    "macro_pca_5": 0.0,
                })
        pd.DataFrame(records).to_parquet(api_module.TEST_FEAT_PATH, index=False)

    # Set mock models in api_module if not present
    if not api_module.models:
        api_module.models["p10"] = DummyModel(0.9)
        api_module.models["p50"] = DummyModel(1.0)
        api_module.models["p90"] = DummyModel(1.1)

    load_resources()

@pytest.fixture(scope="module")
def client(setup_test_environment):
    with TestClient(app) as c:
        yield c


def test_price_bands_endpoint(client):
    # Test with valid parameters (Rice and Punjab exist in test features)
    response = client.get("/api/v1/price-bands?sku_name=Rice&state=Punjab")
    assert response.status_code == 200
    
    data = response.json()
    assert data["sku_name"].lower() == "rice"
    assert data["state"].lower() == "punjab"
    assert "latest_observation_date" in data
    assert "markets" in data
    assert len(data["markets"]) > 0
    
    # Check first market details
    market = data["markets"][0]
    assert "market_mandi" in market
    assert "observed_price_per_quintal" in market
    assert "p10_floor" in market
    assert "p50_midpoint" in market
    assert "p90_ceiling" in market
    assert "compliance_status" in market
    assert market["compliance_status"] in ["WITHIN_BAND", "CEILING_BREACHED"]

def test_price_bands_not_found(client):
    # Test with non-existent SKU/region
    response = client.get("/api/v1/price-bands?sku_name=NonExistent&state=Punjab")
    assert response.status_code == 404
    
    response = client.get("/api/v1/price-bands?sku_name=Rice&state=NonExistent")
    assert response.status_code == 404

def test_monitoring_endpoint(client):
    # Test the live monitoring full table endpoint
    response = client.get("/api/v1/monitoring")
    assert response.status_code == 200
    
    records = response.json()
    assert isinstance(records, list)
    assert len(records) > 0
    
    # Check fields in first record
    rec = records[0]
    assert "sku_name" in rec
    assert "state" in rec
    assert "market_mandi" in rec
    assert "vendor_id" in rec
    assert "observed_price" in rec
    assert "p10_floor" in rec
    assert "p50_mid" in rec
    assert "p90_ceiling" in rec
    assert "compliance_status" in rec


def test_anomalies_endpoint(client):
    # Test the live anomaly detection endpoint
    response = client.get("/api/v1/anomalies")
    assert response.status_code == 200
    
    anomalies = response.json()
    assert isinstance(anomalies, list)
    
    if len(anomalies) > 0:
        a = anomalies[0]
        assert "observation_id" in a
        assert "date" in a
        assert "sku_name" in a
        assert "anomaly_type" in a
        assert "severity_score" in a
        assert "observed_price" in a

    # Test filtering by SKU
    resp_sku = client.get("/api/v1/anomalies?sku_name=Rice")
    assert resp_sku.status_code == 200
    for item in resp_sku.json():
        assert item["sku_name"].lower() == "rice"

