import os
import pytest
import pandas as pd
from unittest.mock import patch

from src.data.vendor_registry import generate_vendor_registry, save_vendor_registry
from src.data.reference_data import get_mrp_ceilings, get_historical_price_controls, get_international_benchmarks
from src.data.temporal_split import perform_temporal_split

def test_vendor_registry():
    # Test generation
    df = generate_vendor_registry(n_vendors=25)
    assert len(df) == 25
    assert list(df.columns) == ["vendor_id", "licensed_status", "retailer_type", "region", "registered_skus"]
    assert df["vendor_id"].iloc[0] == "VEND_0001"
    assert df["vendor_id"].iloc[-1] == "VEND_0025"
    assert df["licensed_status"].dtype == bool
    assert df["retailer_type"].isin(["wholesaler", "retail_chain", "independent"]).all()

def test_save_vendor_registry(tmp_path):
    df = generate_vendor_registry(n_vendors=10)
    out_file = os.path.join(tmp_path, "vendors.parquet")
    save_vendor_registry(df, out_file)
    assert os.path.exists(out_file)
    df_read = pd.read_parquet(out_file)
    assert len(df_read) == 10

def test_reference_data():
    mrp_df = get_mrp_ceilings()
    assert "sku_name" in mrp_df.columns
    assert "mrp_price_per_kg" in mrp_df.columns
    
    pco_df = get_historical_price_controls()
    assert "order_id" in pco_df.columns
    assert "price_cap_per_quintal" in pco_df.columns

    bench_df = get_international_benchmarks()
    assert "observation_date" in bench_df.columns
    assert "fao_food_price_index" in bench_df.columns
    assert len(bench_df) > 0

def test_temporal_split(tmp_path):
    # Create mock raw data spanning 10 unique dates
    dates = pd.date_range(start="2026-01-01", end="2026-01-10", freq="D")
    raw_records = []
    
    # 2 records per day
    for dt in dates:
        date_str = dt.strftime("%Y-%m-%d")
        raw_records.append({"observation_date": date_str, "market_mandi": "Mandi A", "modal_price_per_quintal": 2000.0})
        raw_records.append({"observation_date": date_str, "market_mandi": "Mandi B", "modal_price_per_quintal": 2200.0})
        
    df_raw = pd.DataFrame(raw_records)
    
    # Write to temp input directory (mock partitioned agmarknet folder)
    input_dir = os.path.join(tmp_path, "raw_agmarknet")
    os.makedirs(input_dir, exist_ok=True)
    df_raw.to_parquet(os.path.join(input_dir, "data.parquet"), index=False)
    
    output_dir = os.path.join(tmp_path, "processed")
    
    # Run split
    perform_temporal_split(input_dir, output_dir)
    
    # Verify split output files exist
    assert os.path.exists(os.path.join(output_dir, "train.parquet"))
    assert os.path.exists(os.path.join(output_dir, "val.parquet"))
    assert os.path.exists(os.path.join(output_dir, "test.parquet"))
    
    # Read split files back
    df_train = pd.read_parquet(os.path.join(output_dir, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(output_dir, "val.parquet"))
    df_test = pd.read_parquet(os.path.join(output_dir, "test.parquet"))
    
    # Ensure correct proportions: 6 days train, 2 days val, 2 days test
    # (Since we generated 2 records per day, counts should be 12, 4, 4)
    assert len(df_train) == 12
    assert len(df_val) == 4
    assert len(df_test) == 4
    
    # Verify no overlaps and strict chronological sorting
    train_dates = pd.to_datetime(df_train["observation_date"]).unique()
    val_dates = pd.to_datetime(df_val["observation_date"]).unique()
    test_dates = pd.to_datetime(df_test["observation_date"]).unique()
    
    assert max(train_dates) < min(val_dates)
    assert max(val_dates) < min(test_dates)
