import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("reference_data")

REF_OUT_DIR = "data/raw/reference"

def get_mrp_ceilings() -> pd.DataFrame:
    """
    Returns the statutory Maximum Retail Price (MRP) ceilings for commodities.
    """
    logger.info("Initializing statutory MRP ceilings...")
    mrp_data = [
        {"sku_name": "Rice", "mrp_price_per_kg": 65.0, "effective_from": "2024-01-01", "regulatory_authority": "Department of Consumer Affairs"},
        {"sku_name": "Wheat", "mrp_price_per_kg": 35.0, "effective_from": "2024-01-01", "regulatory_authority": "Department of Consumer Affairs"},
        {"sku_name": "Potato", "mrp_price_per_kg": 30.0, "effective_from": "2024-01-01", "regulatory_authority": "State Civil Supplies Dept"},
        {"sku_name": "Onion", "mrp_price_per_kg": 50.0, "effective_from": "2024-01-01", "regulatory_authority": "State Civil Supplies Dept"},
        {"sku_name": "Tomato", "mrp_price_per_kg": 40.0, "effective_from": "2024-01-01", "regulatory_authority": "State Civil Supplies Dept"},
        {"sku_name": "Pulses", "mrp_price_per_kg": 120.0, "effective_from": "2024-01-01", "regulatory_authority": "Department of Consumer Affairs"}
    ]
    return pd.DataFrame(mrp_data)

def get_historical_price_controls() -> pd.DataFrame:
    """
    Returns historical government price control orders.
    """
    logger.info("Initializing historical government price control orders...")
    price_controls = [
        {"order_id": "PCO_2024_001", "issue_date": "2024-04-15", "commodity": "Onion", "region": "Maharashtra", "price_cap_per_quintal": 4500.0, "status": "Expired"},
        {"order_id": "PCO_2024_002", "issue_date": "2024-08-01", "commodity": "Potato", "region": "Uttar Pradesh", "price_cap_per_quintal": 2500.0, "status": "Active"},
        {"order_id": "PCO_2025_001", "issue_date": "2025-01-10", "commodity": "Tomato", "region": "Karnataka", "price_cap_per_quintal": 3000.0, "status": "Active"},
        {"order_id": "PCO_2026_001", "issue_date": "2026-06-15", "commodity": "Wheat", "region": "Punjab", "price_cap_per_quintal": 2350.0, "status": "Active"}
    ]
    return pd.DataFrame(price_controls)

def get_international_benchmarks() -> pd.DataFrame:
    """
    Returns international benchmark price feeds (FAO/WHO/IEA monthly indicators).
    """
    logger.info("Initializing international benchmark indices...")
    # Generate monthly indices from Jan 2020 to Aug 2026
    date_range = pd.date_range(start="2020-01-01", end="2026-08-01", freq="MS")
    records = []
    
    # FAO Food Price Index (averages 100-140)
    # WHO Health indicator (proxy index, averages 90-110)
    # IEA Energy Index (averages 80-150)
    for idx, dt in enumerate(date_range):
        fao = round(110.0 + (idx * 0.4) + (10.0 * (idx % 12 == 11)), 2)
        who = round(100.0 + (idx * 0.1), 2)
        iea = round(95.0 + (idx * 0.6) - (15.0 * (dt.year == 2020)), 2)  # dip in 2020
        records.append({
            "observation_date": dt.strftime("%Y-%m-%d"),
            "fao_food_price_index": fao,
            "who_health_indicator": who,
            "iea_energy_index": iea
        })
    return pd.DataFrame(records)

def save_reference_data(output_dir: str = REF_OUT_DIR) -> None:
    """
    Initializes and saves reference datasets to Parquet files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # MRP Ceilings
    mrp_df = get_mrp_ceilings()
    mrp_path = os.path.join(output_dir, "mrp_ceilings.parquet")
    mrp_df.to_parquet(mrp_path, compression="snappy", index=False)
    logger.info("Saved MRP ceilings to %s", mrp_path)

    # Price Control Orders
    pco_df = get_historical_price_controls()
    pco_path = os.path.join(output_dir, "price_control_orders.parquet")
    pco_df.to_parquet(pco_path, compression="snappy", index=False)
    logger.info("Saved historical price controls to %s", pco_path)

    # International Benchmarks
    bench_df = get_international_benchmarks()
    bench_path = os.path.join(output_dir, "international_benchmarks.parquet")
    bench_df.to_parquet(bench_path, compression="snappy", index=False)
    logger.info("Saved international benchmarks to %s", bench_path)

def main():
    parser = argparse.ArgumentParser(description="Statutory Reference Data Initializer for CASPER-Gov")
    parser.add_argument("--output-dir", type=str, default=REF_OUT_DIR, help="Directory to save Parquet reference tables")
    args = parser.parse_args()
    
    save_reference_data(args.output_dir)

if __name__ == "__main__":
    main()
