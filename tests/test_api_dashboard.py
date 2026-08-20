import os
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, load_resources

@pytest.fixture(scope="module")
def client():
    # Make sure startup events run (which load models, clusters, and vendors)
    load_resources()
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
