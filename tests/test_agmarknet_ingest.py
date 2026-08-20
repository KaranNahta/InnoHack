import os
import pytest
import pandas as pd
from unittest.mock import patch

from src.data.agmarknet_ingest import (
    fetch_daily_mandi_prices,
    normalize_agmarknet_schema,
    validate_agmarknet_schema,
    save_raw_agmarknet_data
)

@pytest.fixture
def mock_raw_data():
    return [
        {
            "state_name": "Uttar Pradesh",
            "district_name": "Agra",
            "market_name": "Agra Mandi",
            "commodity": "Rice",
            "variety": "Basmati",
            "min_price": 3000.0,
            "max_price": 3500.0,
            "modal_price": 3200.0,
            "arrivals": 100.0,
            "reported_date": "2026-01-01"
        },
        {
            "state_name": "Uttar Pradesh",
            "district_name": "Agra",
            "market_name": "Agra Mandi",
            "commodity": "Rice",
            "variety": "Basmati",
            "min_price": 3000.0,
            "max_price": 3500.0,
            # Test zero imputation: should be imputed to 3200.0 (the weekly median for the group)
            "modal_price": 0.0,
            "arrivals": 50.0,
            "reported_date": "2026-01-02"
        },
        {
            "state_name": "Uttar Pradesh",
            "district_name": "Agra",
            "market_name": "Agra Mandi",
            "commodity": "Rice",
            "variety": "Basmati",
            "min_price": 3000.0,
            "max_price": 3500.0,
            # Test missing value imputation: should be imputed to 3200.0
            "modal_price": None,
            "arrivals": 80.0,
            "reported_date": "2026-01-03"
        }
    ]

@patch("src.data.agmarknet_ingest.AgmarknetClient")
def test_fetch_daily_mandi_prices(MockClient, mock_raw_data):
    mock_instance = MockClient.return_value
    mock_instance.fetch_raw_data.return_value = mock_raw_data

    df = fetch_daily_mandi_prices("2026-01-01", "2026-01-03", ["Rice"], ["Uttar Pradesh"])
    assert len(df) == 3
    assert df.iloc[0]["market_name"] == "Agra Mandi"
    assert df.iloc[1]["modal_price"] == 0.0
    assert pd.isna(df.iloc[2]["modal_price"])

def test_normalize_agmarknet_schema(mock_raw_data):
    df_raw = pd.DataFrame(mock_raw_data)
    df_norm = normalize_agmarknet_schema(df_raw)

    # Check renamed columns
    assert "state" in df_norm.columns
    assert "district" in df_norm.columns
    assert "market_mandi" in df_norm.columns
    assert "sku_name" in df_norm.columns
    assert "sku_variety" in df_norm.columns
    assert "min_price_per_quintal" in df_norm.columns
    assert "max_price_per_quintal" in df_norm.columns
    assert "modal_price_per_quintal" in df_norm.columns
    assert "arrival_quantity_tonnes" in df_norm.columns
    assert "observation_date" in df_norm.columns
    assert "modal_price_per_kg" in df_norm.columns

    # Check datatype standardization
    assert pd.api.types.is_datetime64_any_dtype(df_norm["observation_date"])

    # Check arrival conversion (100.0 / 10.0 = 10.0 tonnes)
    assert df_norm.iloc[0]["arrival_quantity_tonnes"] == 10.0

    # Check price conversion (3200.0 / 100.0 = 32.0 per kg)
    assert df_norm.iloc[0]["modal_price_per_kg"] == 32.0

    # Check imputation (zero and None should be replaced by the weekly median of 3200.0)
    assert df_norm.iloc[1]["modal_price_per_quintal"] == 3200.0
    assert df_norm.iloc[2]["modal_price_per_quintal"] == 3200.0
    assert df_norm.iloc[1]["modal_price_per_kg"] == 32.0
    assert df_norm.iloc[2]["modal_price_per_kg"] == 32.0

def test_validate_agmarknet_schema(mock_raw_data):
    df_raw = pd.DataFrame(mock_raw_data)
    df_norm = normalize_agmarknet_schema(df_raw)
    
    # Should validate successfully
    df_val = validate_agmarknet_schema(df_norm)
    assert len(df_val) == 3

    # Introduce validation failure (e.g. state is missing)
    df_bad = df_norm.copy()
    # Cast column to object to allow string/None mixed values
    df_bad["state"] = df_bad["state"].astype(object)
    df_bad.loc[0, "state"] = None
    
    df_val_bad = validate_agmarknet_schema(df_bad)
    # The first row is dropped/filtered out due to validation failure
    assert len(df_val_bad) == 2

def test_save_raw_agmarknet_data(tmp_path, mock_raw_data):
    df_raw = pd.DataFrame(mock_raw_data)
    df_norm = normalize_agmarknet_schema(df_raw)
    df_val = validate_agmarknet_schema(df_norm)

    output_dir = os.path.join(tmp_path, "agmarknet_raw")
    save_raw_agmarknet_data(df_val, output_dir)

    # Check partitioned folders exist
    # Dates are all in 2026-01
    partition_path = os.path.join(output_dir, "year=2026", "month=1")
    assert os.path.exists(partition_path)

    # Check we can read it back
    df_read = pd.read_parquet(output_dir)
    assert len(df_read) == 3
    # Check partition columns are retrieved by pandas
    assert "year" in df_read.columns
    assert "month" in df_read.columns
    assert (df_read["year"] == 2026).all()
    assert (df_read["month"] == 1).all()
