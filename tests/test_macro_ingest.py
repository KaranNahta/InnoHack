import os
import pytest
import pandas as pd
import pyarrow as pa
from unittest.mock import patch, MagicMock
from src.data.macro_ingest import (
    ensure_fallback_files,
    ingest_cpi,
    ingest_wpi,
    validate_and_save,
    run_macro_ingest,
    CPI_SCHEMA
)

@pytest.fixture
def mock_world_bank_response():
    return [
        {"page": 1, "pages": 1, "per_page": 50, "total": 2},
        [
            {
                "indicator": {"id": "FP.CPI.TOTL", "value": "Consumer price index (2010 = 100)"},
                "country": {"id": "IN", "value": "India"},
                "countryiso3code": "IND",
                "date": "2020",
                "value": 150.5,
                "unit": "",
                "obs_status": "",
                "decimal": 1
            },
            {
                "indicator": {"id": "FP.CPI.TOTL", "value": "Consumer price index (2010 = 100)"},
                "country": {"id": "IN", "value": "India"},
                "countryiso3code": "IND",
                "date": "2021",
                "value": 158.2,
                "unit": "",
                "obs_status": "",
                "decimal": 1
            }
        ]
    ]

def test_ensure_fallback_files(tmp_path):
    # Temporarily patch FALLBACK_DIR
    with patch("src.data.macro_ingest.FALLBACK_DIR", str(tmp_path)):
        ensure_fallback_files()
        assert os.path.exists(os.path.join(tmp_path, "cpi.csv"))
        assert os.path.exists(os.path.join(tmp_path, "wpi.csv"))
        assert os.path.exists(os.path.join(tmp_path, "freight.csv"))
        assert os.path.exists(os.path.join(tmp_path, "harvest.csv"))

        df_cpi = pd.read_csv(os.path.join(tmp_path, "cpi.csv"))
        assert "observation_date" in df_cpi.columns
        assert "cpi_value" in df_cpi.columns

@patch("src.data.macro_ingest.requests.get")
def test_ingest_cpi_world_bank(mock_get, mock_world_bank_response):
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = mock_world_bank_response
    mock_get.return_value = mock_res

    df = ingest_cpi()
    assert len(df) == 2
    assert df.iloc[0]["cpi_value"] == 150.5
    assert df.iloc[0]["series_id"] == "FP.CPI.TOTL"
    assert df.iloc[0]["observation_date"] == "2020-12-31"

def test_validate_and_save_success(tmp_path):
    df = pd.DataFrame({
        "observation_date": ["2026-01-01", "2026-02-01"],
        "cpi_value": [120.5, 121.2],
        "series_id": ["TEST_SERIES", "TEST_SERIES"]
    })
    
    with patch("src.data.macro_ingest.RAW_OUT_DIR", str(tmp_path)):
        validate_and_save(df, CPI_SCHEMA, "test_cpi.parquet")
        out_file = os.path.join(tmp_path, "test_cpi.parquet")
        assert os.path.exists(out_file)

        # Read it back and verify schema
        table = pa.parquet.read_table(out_file)
        assert table.schema.equals(CPI_SCHEMA)

def test_validate_and_save_invalid_schema(tmp_path):
    # Missing required column 'series_id'
    df = pd.DataFrame({
        "observation_date": ["2026-01-01"],
        "cpi_value": [120.5]
    })
    with patch("src.data.macro_ingest.RAW_OUT_DIR", str(tmp_path)):
        with pytest.raises(KeyError):
            validate_and_save(df, CPI_SCHEMA, "test_invalid.parquet")

def test_run_macro_ingest_all(tmp_path):
    # Patch both fallback and output directories to use temp paths
    fallback_dir = os.path.join(tmp_path, "fallback")
    output_dir = os.path.join(tmp_path, "raw")
    
    with patch("src.data.macro_ingest.FALLBACK_DIR", fallback_dir), \
         patch("src.data.macro_ingest.RAW_OUT_DIR", output_dir), \
         patch("src.data.macro_ingest.requests.get") as mock_get:
        
        # Force requests to fail so it uses fallback
        mock_get.side_effect = Exception("Network down")
        
        run_macro_ingest(["all"])
        
        assert os.path.exists(os.path.join(output_dir, "cpi.parquet"))
        assert os.path.exists(os.path.join(output_dir, "wpi.parquet"))
        assert os.path.exists(os.path.join(output_dir, "freight.parquet"))
        assert os.path.exists(os.path.join(output_dir, "harvest.parquet"))
