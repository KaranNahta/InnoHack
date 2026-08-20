import pytest
from fastapi.testclient import TestClient
from src.api.main import app, load_resources

@pytest.fixture(scope="module")
def client():
    # Make sure models and metadata are initialized
    load_resources()
    return TestClient(app)

def test_risk_analysis_upload_success(client):
    # Construct a valid mock CSV payload
    csv_payload = (
        "sku_name,state,district,market_mandi,sku_variety,observed_price\n"
        "Tomato,Uttar Pradesh,Varanasi,Varanasi Mandi,Desi,9500.0\n"   # Extreme high -> HIGH RISK
        "Tomato,Uttar Pradesh,Varanasi,Varanasi Mandi,Desi,10.0\n"     # Extreme low -> LOW RISK
    )
    
    response = client.post(
        "/api/v1/risk-analysis",
        files={"file": ("test_transactions.csv", csv_payload, "text/csv")}
    )
    
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 2
    
    # Assert schema
    for record in records:
        assert "sku_name" in record
        assert "state" in record
        assert "observed_price" in record
        assert "p10_floor" in record
        assert "p50_midpoint" in record
        assert "p90_ceiling" in record
        assert "risk_rating" in record
        
    # High price matches HIGH RISK
    assert records[0]["risk_rating"] == "HIGH RISK"
    # Low price matches LOW RISK
    assert records[1]["risk_rating"] == "LOW RISK"

def test_risk_analysis_upload_missing_columns(client):
    # CSV missing observed_price
    invalid_csv = (
        "sku_name,state,district,market_mandi,sku_variety\n"
        "Tomato,Uttar Pradesh,Varanasi,Varanasi Mandi,Desi\n"
    )
    
    response = client.post(
        "/api/v1/risk-analysis",
        files={"file": ("test_invalid.csv", invalid_csv, "text/csv")}
    )
    
    assert response.status_code == 400
    assert "observed_price" in response.json()["detail"]

def test_risk_analysis_upload_invalid_file(client):
    # Non-CSV binary data
    response = client.post(
        "/api/v1/risk-analysis",
        files={"file": ("test_binary.bin", b"\x00\x01\x02\x03", "application/octet-stream")}
    )
    # Binary upload parsed as string might not contain columns
    assert response.status_code == 400
