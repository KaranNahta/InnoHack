import os
import pytest
import pandas as pd
import numpy as np
import chromadb
from pydantic import BaseModel

from src.models.shap_explainer import explain_price_anomaly
from src.rag.vector_store import populate_database, query_precedents
from src.llm.report_generator import generate_enforcement_notice, EnforcementNotice

@pytest.fixture
def mock_explain_row():
    # Create single-row DataFrame with expected feature columns
    row = pd.DataFrame([{
        "price_lag_7d": 3200.0, "price_lag_14d": 3100.0, "price_lag_30d": 3000.0, "price_lag_90d": 2900.0,
        "volatility_7d": 0.05, "volatility_30d": 0.08, "seasonal_index": 1.05, "supply_shock_zscore": -1.5, "is_harvest_season": 0.0,
        "macro_pca_1": -0.8, "macro_pca_2": 0.2, "macro_pca_3": 0.1, "macro_pca_4": -0.3, "macro_pca_5": 0.0,
        "sku_name": "Tomato", "state": "Uttar Pradesh", "district": "Varanasi", "market_mandi": "Varanasi Mandi", "sku_variety": "Desi", "cluster_id": "-1"
    }])
    return row

def test_shap_explanation(mock_explain_row):
    # Verify that the SHAP explainer extracts cost drivers correctly
    # Test requires the fitted lgb_p50 model to exist in models/
    model_path = "models/lgb_p50.joblib"
    if os.path.exists(model_path):
        drivers = explain_price_anomaly(mock_explain_row, model_path=model_path)
        assert isinstance(drivers, list)
        assert len(drivers) == 5
        
        # Verify schema
        for driver in drivers:
            assert "feature" in driver
            assert "contribution_percentage" in driver
            assert "impact_direction" in driver
            assert isinstance(driver["contribution_percentage"], float)
            assert driver["impact_direction"] in ["INCREASE", "DECREASE"]
            
        # Top driver should have highest percentage
        assert drivers[0]["contribution_percentage"] >= drivers[1]["contribution_percentage"]
    else:
        pytest.skip("lgb_p50.joblib model not found, skipping explanation test.")

def test_vector_store_rag(tmp_path):
    db_path = os.path.join(tmp_path, "chroma")
    
    # 1. Populate database
    populate_database(db_path=db_path)
    assert os.path.exists(os.path.join(db_path, "chroma.sqlite3"))
    
    # 2. Query precedents
    docs = query_precedents("Section 3 Essential Commodities Act penalties", n_results=1, db_path=db_path)
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert "Essential Commodities Act" in docs[0]

def test_report_generator():
    anomaly = {
        "date": "2026-08-20",
        "sku_name": "Tomato",
        "state": "Uttar Pradesh",
        "anomaly_type": "PRICE_GOUGING_ALERT",
        "severity_score": 0.85,
        "vendors_involved": ["VEND_0110"]
    }
    
    drivers = [
        {"feature": "Mandi Arrival Supply Shock", "contribution_percentage": 52.1, "impact_direction": "INCREASE"},
        {"feature": "Recent 7-Day Price Lag", "contribution_percentage": 31.4, "impact_direction": "INCREASE"}
    ]
    
    precedents = [
        "Essential Commodities Act (Section 3) - Section 3 allows penalties.",
        "Regulatory Intervention Policy - Price warnings triggered."
    ]
    
    notice = generate_enforcement_notice(anomaly, drivers, precedents)
    
    # Verify outputs against Pydantic schema
    assert isinstance(notice, EnforcementNotice)
    assert notice.severity_rating == "CRITICAL"
    assert "Tomato" in notice.probable_cause
    assert len(notice.top_cost_drivers) == 2
    assert "Mandi Arrival Supply Shock" == notice.top_cost_drivers[0].feature
    assert "Essential Commodities Act" in notice.draft_enforcement_notice_text
    assert "WARNING" in notice.draft_enforcement_notice_text.upper()
