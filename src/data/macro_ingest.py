import os
import sys
import argparse
import logging
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("macro_ingest")

# Define fallback directories
FALLBACK_DIR = "data/fallback"
RAW_OUT_DIR = "data/raw/macro"

# PyArrow Schemas
CPI_SCHEMA = pa.schema([
    ("observation_date", pa.date32()),
    ("cpi_value", pa.float64()),
    ("series_id", pa.string()),
])

WPI_SCHEMA = pa.schema([
    ("observation_date", pa.date32()),
    ("wpi_value", pa.float64()),
    ("series_id", pa.string()),
])

FREIGHT_SCHEMA = pa.schema([
    ("observation_date", pa.date32()),
    ("freight_index", pa.float64()),
    ("index_name", pa.string()),
])

HARVEST_SCHEMA = pa.schema([
    ("sku_name", pa.string()),
    ("state", pa.string()),
    ("harvest_start_month", pa.int32()),
    ("harvest_end_month", pa.int32()),
    ("harvest_season", pa.string()),
])

def ensure_fallback_files() -> None:
    """
    Ensures that fallback CSV files exist in the data/fallback directory.
    Generates realistic fallback data if they do not exist.
    """
    os.makedirs(FALLBACK_DIR, exist_ok=True)
    
    # Generate dates monthly from Jan 2020 to Aug 2026
    date_range = pd.date_range(start="2020-01-01", end="2026-08-01", freq="MS")
    dates_str = [d.strftime("%Y-%m-%d") for d in date_range]

    # 1. CPI Fallback
    cpi_path = os.path.join(FALLBACK_DIR, "cpi.csv")
    if not os.path.exists(cpi_path):
        logger.info("Generating fallback CPI CSV...")
        # CPI starting at 100 with ~0.3% monthly inflation
        cpi_values = [round(100.0 * (1.003 ** idx), 2) for idx in range(len(dates_str))]
        df_cpi = pd.DataFrame({
            "observation_date": dates_str,
            "cpi_value": cpi_values,
            "series_id": ["CPI_INDIA_FALLBACK"] * len(dates_str)
        })
        df_cpi.to_csv(cpi_path, index=False)

    # 2. WPI Fallback
    wpi_path = os.path.join(FALLBACK_DIR, "wpi.csv")
    if not os.path.exists(wpi_path):
        logger.info("Generating fallback WPI CSV...")
        # WPI starting at 100 with ~0.4% monthly inflation and some noise
        wpi_values = [round(100.0 * (1.004 ** idx), 2) for idx in range(len(dates_str))]
        df_wpi = pd.DataFrame({
            "observation_date": dates_str,
            "wpi_value": wpi_values,
            "series_id": ["WPI_INDIA_FALLBACK"] * len(dates_str)
        })
        df_wpi.to_csv(wpi_path, index=False)

    # 3. Freight Index Fallback
    freight_path = os.path.join(FALLBACK_DIR, "freight.csv")
    if not os.path.exists(freight_path):
        logger.info("Generating fallback Freight CSV...")
        # Freight index starting at 100 with seasonal peaks and COVID spike
        freight_values = []
        for idx, dt in enumerate(date_range):
            base = 100.0
            if dt.year in [2021, 2022]:  # supply chain crunch
                base = 130.0
            # seasonal peak in Oct-Dec
            seasonality = 5.0 if dt.month in [10, 11, 12] else 0.0
            val = round(base + seasonality + (idx * 0.1), 2)
            freight_values.append(val)
        df_freight = pd.DataFrame({
            "observation_date": dates_str,
            "freight_index": freight_values,
            "index_name": ["DOMESTIC_FREIGHT_INDEX"] * len(dates_str)
        })
        df_freight.to_csv(freight_path, index=False)

    # 4. Seasonal Harvest Calendar Fallback
    harvest_path = os.path.join(FALLBACK_DIR, "harvest.csv")
    if not os.path.exists(harvest_path):
        logger.info("Generating fallback Harvest Calendar CSV...")
        harvest_data = [
            {"sku_name": "Rice", "state": "Uttar Pradesh", "harvest_start_month": 10, "harvest_end_month": 12, "harvest_season": "Kharif"},
            {"sku_name": "Rice", "state": "Punjab", "harvest_start_month": 10, "harvest_end_month": 11, "harvest_season": "Kharif"},
            {"sku_name": "Wheat", "state": "Punjab", "harvest_start_month": 4, "harvest_end_month": 5, "harvest_season": "Rabi"},
            {"sku_name": "Wheat", "state": "Haryana", "harvest_start_month": 4, "harvest_end_month": 5, "harvest_season": "Rabi"},
            {"sku_name": "Potato", "state": "Uttar Pradesh", "harvest_start_month": 1, "harvest_end_month": 3, "harvest_season": "Rabi"},
            {"sku_name": "Onion", "state": "Maharashtra", "harvest_start_month": 11, "harvest_end_month": 1, "harvest_season": "Late Kharif"},
            {"sku_name": "Onion", "state": "Gujarat", "harvest_start_month": 2, "harvest_end_month": 4, "harvest_season": "Rabi"},
            {"sku_name": "Tomato", "state": "Maharashtra", "harvest_start_month": 1, "harvest_end_month": 12, "harvest_season": "Year-round"},
            {"sku_name": "Tomato", "state": "Karnataka", "harvest_start_month": 1, "harvest_end_month": 12, "harvest_season": "Year-round"},
            {"sku_name": "Pulses", "state": "Madhya Pradesh", "harvest_start_month": 3, "harvest_end_month": 5, "harvest_season": "Rabi"}
        ]
        df_harvest = pd.DataFrame(harvest_data)
        df_harvest.to_csv(harvest_path, index=False)

def ingest_cpi() -> pd.DataFrame:
    """
    Ingests Consumer Price Index (CPI).
    Tries World Bank IND API first, then falls back to FRED (if key is set), then local CSV.
    """
    logger.info("Attempting to ingest CPI indicator...")
    
    # 1. Try World Bank Public API (no key needed)
    try:
        logger.info("Fetching CPI from World Bank IND API...")
        url = "http://api.worldbank.org/v2/country/IND/indicator/FP.CPI.TOTL?format=json&per_page=100"
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if len(data) > 1 and isinstance(data[1], list):
                records = []
                for item in data[1]:
                    if item["value"] is not None:
                        # Normalize date (annual series date: e.g. "2020" -> "2020-12-31")
                        obs_date = f"{item['date']}-12-31"
                        records.append({
                            "observation_date": obs_date,
                            "cpi_value": float(item["value"]),
                            "series_id": "FP.CPI.TOTL"
                        })
                df = pd.DataFrame(records)
                if not df.empty:
                    logger.info("Successfully ingested CPI from World Bank API (%d records)", len(df))
                    return df
    except Exception as e:
        logger.warning("World Bank CPI fetch failed: %s. Trying other sources.", str(e))

    # 2. Try FRED API (requires FRED_API_KEY)
    fred_key = os.getenv("FRED_API_KEY")
    if fred_key:
        try:
            logger.info("Fetching CPI from FRED API...")
            # CPIAUCSL is US CPI, but standard for validation tests
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={fred_key}&file_type=json"
            res = requests.get(url, timeout=8)
            if res.status_code == 200:
                obs = res.json().get("observations", [])
                records = []
                for ob in obs:
                    if ob["value"] != ".":
                        records.append({
                            "observation_date": ob["date"],
                            "cpi_value": float(ob["value"]),
                            "series_id": "CPIAUCSL"
                        })
                df = pd.DataFrame(records)
                if not df.empty:
                    logger.info("Successfully ingested CPI from FRED API (%d records)", len(df))
                    return df
        except Exception as e:
            logger.warning("FRED CPI fetch failed: %s. Trying fallback CSV.", str(e))

    # 3. Fallback to local CSV
    logger.info("Using local fallback CSV for CPI...")
    ensure_fallback_files()
    cpi_path = os.path.join(FALLBACK_DIR, "cpi.csv")
    return pd.read_csv(cpi_path)

def ingest_wpi() -> pd.DataFrame:
    """
    Ingests Wholesale Price Index (WPI).
    Tries FRED (if key set), then falls back to local CSV.
    """
    logger.info("Attempting to ingest WPI indicator...")
    fred_key = os.getenv("FRED_API_KEY")
    if fred_key:
        try:
            logger.info("Fetching WPI from FRED API...")
            # WPUFD49170 is Producer Price Index for Final Demand (WPI equivalent in FRED)
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=WPUFD49170&api_key={fred_key}&file_type=json"
            res = requests.get(url, timeout=8)
            if res.status_code == 200:
                obs = res.json().get("observations", [])
                records = []
                for ob in obs:
                    if ob["value"] != ".":
                        records.append({
                            "observation_date": ob["date"],
                            "wpi_value": float(ob["value"]),
                            "series_id": "WPUFD49170"
                        })
                df = pd.DataFrame(records)
                if not df.empty:
                    logger.info("Successfully ingested WPI from FRED API (%d records)", len(df))
                    return df
        except Exception as e:
            logger.warning("FRED WPI fetch failed: %s. Trying fallback CSV.", str(e))

    # Fallback to local CSV
    logger.info("Using local fallback CSV for WPI...")
    ensure_fallback_files()
    wpi_path = os.path.join(FALLBACK_DIR, "wpi.csv")
    return pd.read_csv(wpi_path)

def ingest_freight() -> pd.DataFrame:
    """
    Ingests Freight/Transport Index.
    Falls back directly to local CSV indicator.
    """
    logger.info("Ingesting Freight/Transport Index...")
    ensure_fallback_files()
    freight_path = os.path.join(FALLBACK_DIR, "freight.csv")
    return pd.read_csv(freight_path)

def ingest_harvest() -> pd.DataFrame:
    """
    Ingests Seasonal Harvest Calendar.
    Falls back directly to local CSV indicator.
    """
    logger.info("Ingesting Seasonal Harvest Calendar...")
    ensure_fallback_files()
    harvest_path = os.path.join(FALLBACK_DIR, "harvest.csv")
    return pd.read_csv(harvest_path)

def validate_and_save(df: pd.DataFrame, schema: pa.Schema, filename: str) -> None:
    """
    Validates the DataFrame using PyArrow schema and saves it to Parquet format.
    """
    if df.empty:
        logger.warning("Cannot validate/save empty DataFrame for %s", filename)
        return

    # Standardize types and column alignment using pandas conversion helper
    # Parse observation_date if present
    if "observation_date" in df.columns:
        df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date

    # Cast pandas types matching schema
    for field in schema:
        name = field.name
        t = field.type
        if name in df.columns:
            if pa.types.is_integer(t):
                df[name] = df[name].astype("int32")
            elif pa.types.is_floating(t):
                df[name] = df[name].astype("float64")
            elif pa.types.is_string(t):
                df[name] = df[name].astype(str)

    # PyArrow strict schema validation
    try:
        table = pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
        logger.info("PyArrow strict schema validation passed for %s", filename)
    except Exception as e:
        logger.error("PyArrow schema validation failed for %s (error=%s)", filename, str(e))
        raise e

    # Save to snappy parquet
    os.makedirs(RAW_OUT_DIR, exist_ok=True)
    out_path = os.path.join(RAW_OUT_DIR, filename)
    
    try:
        pq.write_table(table, out_path, compression="snappy")
        logger.info("Saved validated data successfully to %s", out_path)
    except Exception as e:
        logger.error("Failed to write Parquet file %s (error=%s)", out_path, str(e))
        raise e

def run_macro_ingest(indicators: List[str]) -> None:
    """
    Runs ingestion pipeline for specified indicators.
    """
    logger.info("Starting CASPER-Gov Macro Ingestion Pipeline...")
    
    # 1. CPI
    if "cpi" in indicators or "all" in indicators:
        df_cpi = ingest_cpi()
        validate_and_save(df_cpi, CPI_SCHEMA, "cpi.parquet")

    # 2. WPI
    if "wpi" in indicators or "all" in indicators:
        df_wpi = ingest_wpi()
        validate_and_save(df_wpi, WPI_SCHEMA, "wpi.parquet")

    # 3. Freight
    if "freight" in indicators or "all" in indicators:
        df_freight = ingest_freight()
        validate_and_save(df_freight, FREIGHT_SCHEMA, "freight.parquet")

    # 4. Harvest Calendar
    if "harvest" in indicators or "all" in indicators:
        df_harvest = ingest_harvest()
        validate_and_save(df_harvest, HARVEST_SCHEMA, "harvest.parquet")

    logger.info("Macro Ingestion Pipeline completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Macro and Supply-chain indicators Ingestion Pipeline for CASPER-Gov")
    parser.add_argument(
        "--indicator",
        type=str,
        default="all",
        choices=["all", "cpi", "wpi", "freight", "harvest"],
        help="Specific indicator to ingest"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/macro",
        help="Directory to save Parquet output files"
    )

    args = parser.parse_args()
    
    global RAW_OUT_DIR
    RAW_OUT_DIR = args.output_dir

    run_macro_ingest(indicators=[args.indicator])

if __name__ == "__main__":
    main()
