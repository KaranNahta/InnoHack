import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from agmarknet_api import AgmarknetClient
from src.config import DEFAULT_COMMODITIES, DEFAULT_STATES

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("agmarknet_ingest")

class MandiRecord(BaseModel):
    state: str
    district: str
    market_mandi: str
    sku_name: str
    sku_variety: str
    min_price_per_quintal: float
    max_price_per_quintal: float
    modal_price_per_quintal: Optional[float] = None
    modal_price_per_kg: Optional[float] = None
    arrival_quantity_tonnes: float
    observation_date: datetime

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        "Transient failure during Agmarknet API fetch, retrying... (attempt: %s, exception: %s)",
        retry_state.attempt_number,
        str(retry_state.outcome.exception())
    )
)
def _fetch_raw_with_retry(client: AgmarknetClient, start_date: str, end_date: str, commodities: List[str], states: List[str]) -> List[Dict[str, Any]]:
    return client.fetch_raw_data(start_date=start_date, end_date=end_date, commodities=commodities, states=states)

def fetch_daily_mandi_prices(
    start_date: str,
    end_date: str,
    commodities: List[str] = None,
    states: List[str] = None
) -> pd.DataFrame:
    """
    Fetches raw daily mandi prices and returns a DataFrame.
    """
    logger.info(
        "Initializing Agmarknet API fetch request (start_date=%s, end_date=%s, commodities=%s, states=%s)",
        start_date, end_date, commodities, states
    )
    
    client = AgmarknetClient()
    try:
        raw_records = _fetch_raw_with_retry(
            client=client,
            start_date=start_date,
            end_date=end_date,
            commodities=commodities,
            states=states
        )
    except Exception as e:
        logger.error("Failed to fetch data from Agmarknet API after retries (error=%s)", str(e))
        raise e

    df = pd.DataFrame(raw_records)
    logger.info("Successfully fetched raw data (raw_records_count=%s)", len(df))
    return df

def normalize_agmarknet_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns to standardized CASPER-Gov schema and normalizes datatypes and values.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to normalization engine")
        return df

    # Renaming mapping
    column_mapping = {
        "state_name": "state",
        "district_name": "district",
        "market_name": "market_mandi",
        "commodity": "sku_name",
        "variety": "sku_variety",
        "min_price": "min_price_per_quintal",
        "max_price": "max_price_per_quintal",
        "modal_price": "modal_price_per_quintal",
        "arrivals": "arrival_quantity_tonnes",
        "reported_date": "observation_date"
    }
    
    # Filter columns to only those present
    mapping_to_apply = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=mapping_to_apply)

    # Standardize observation_date to datetime64[ns]
    df["observation_date"] = pd.to_datetime(df["observation_date"])

    # Convert arrivals from quintals to metric tonnes (1 quintal = 0.1 tonne)
    df["arrival_quantity_tonnes"] = df["arrival_quantity_tonnes"] / 10.0

    # Derive modal_price_per_kg
    df["modal_price_per_kg"] = df["modal_price_per_quintal"] / 100.0

    # Impute missing or zero-value modal prices using market-weekly median
    missing_mask = df["modal_price_per_quintal"].isna() | (df["modal_price_per_quintal"] == 0.0)
    initial_missing_count = missing_mask.sum()
    initial_missing_pct = (initial_missing_count / len(df)) * 100.0 if len(df) > 0 else 0.0
    
    logger.info(
        "Running price anomaly detection and imputation (initial_missing_count=%s, initial_missing_pct=%s%%)",
        int(initial_missing_count), round(initial_missing_pct, 2)
    )

    # Replace 0 or negative values with NaN for median calculation
    df["modal_price_per_quintal"] = df["modal_price_per_quintal"].replace(0.0, np.nan)
    
    # Add temporary year and week columns for grouping
    df["_temp_year"] = df["observation_date"].dt.year
    df["_temp_week"] = df["observation_date"].dt.isocalendar().week

    # Calculate weekly median price per market
    group_cols = ["market_mandi", "_temp_year", "_temp_week"]
    medians = df.groupby(group_cols)["modal_price_per_quintal"].transform("median")
    df["modal_price_per_quintal"] = df["modal_price_per_quintal"].fillna(medians)

    # Re-calculate modal_price_per_kg with imputed values
    df["modal_price_per_kg"] = df["modal_price_per_quintal"] / 100.0

    # Clean up temp columns
    df = df.drop(columns=["_temp_year", "_temp_week"])

    # Identify and log unresolvable anomalies
    unresolved = df[df["modal_price_per_quintal"].isna() | (df["modal_price_per_quintal"] <= 0)]
    final_missing_pct = (len(unresolved) / len(df)) * 100.0 if len(df) > 0 else 0.0

    if not unresolved.empty:
        logger.warning(
            "Detected unresolvable price anomalies after weekly-market median imputation (unresolved_count=%s, unresolved_percentage=%s%%)",
            len(unresolved), round(final_missing_pct, 2)
        )
        for _, row in unresolved.head(10).iterrows():
            logger.warning(
                "Unresolvable modal price details (market=%s, commodity=%s, variety=%s, date=%s)",
                row["market_mandi"], row["sku_name"], row["sku_variety"], row["observation_date"].strftime("%Y-%m-%d")
            )
    else:
        logger.info("Imputation complete. No unresolvable anomalies remain.")

    return df

def validate_agmarknet_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces Pydantic schema validation on all returned records.
    """
    if df.empty:
        return df

    records = df.to_dict(orient="records")
    validated_records = []
    invalid_count = 0

    for idx, r in enumerate(records):
        # Convert Timestamp to Python datetime
        if isinstance(r["observation_date"], pd.Timestamp):
            r["observation_date"] = r["observation_date"].to_pydatetime()
        
        # Convert float types that might be float64/int64
        for k in ["min_price_per_quintal", "max_price_per_quintal", "modal_price_per_quintal", "modal_price_per_kg", "arrival_quantity_tonnes"]:
            if k in r and (isinstance(r[k], (float, int)) or pd.isna(r[k])):
                if pd.isna(r[k]):
                    r[k] = None
                else:
                    r[k] = float(r[k])

        try:
            MandiRecord(**r)
            validated_records.append(r)
        except ValidationError as e:
            invalid_count += 1
            if invalid_count <= 5:
                logger.error("Validation error details (record_index=%s, error=%s)", idx, e.errors())
    
    if invalid_count > 0:
        logger.warning("Pydantic schema validation failures detected (failed_count=%s, total_count=%s)", invalid_count, len(df))
    else:
        logger.info("Pydantic schema validation passed for all records (total_count=%s)", len(df))

    return pd.DataFrame(validated_records)

def save_raw_agmarknet_data(df: pd.DataFrame, output_dir: str = "data/raw/agmarknet") -> None:
    """
    Partitions data by year and month and stores as compressed snappy Parquet files.
    """
    if df.empty:
        logger.warning("No data to save to Parquet")
        return

    # Extract partitioning columns
    df = df.copy()
    df["year"] = df["observation_date"].dt.year
    df["month"] = df["observation_date"].dt.month

    os.makedirs(output_dir, exist_ok=True)
    
    # Write partitioned parquet
    try:
        df.to_parquet(
            output_dir,
            partition_cols=["year", "month"],
            compression="snappy",
            index=False
        )
        logger.info("Saved data successfully in partitioned Parquet format (output_directory=%s)", output_dir)
    except Exception as e:
        logger.error("Failed to write Parquet files (error=%s)", str(e))
        raise e

def run_ingest_pipeline(start_date: str, end_date: str, commodities: List[str] = None, states: List[str] = None, output_dir: str = "data/raw/agmarknet") -> None:
    """
    Main orchestration function running the ingestion pipeline.
    """
    start_time = datetime.now()
    logger.info("Starting CASPER-Gov Agmarknet data ingestion pipeline (start_time=%s)", start_time.isoformat())

    # Step 1: Fetch
    try:
        df_raw = fetch_daily_mandi_prices(start_date, end_date, commodities, states)
    except Exception as e:
        logger.critical("Pipeline aborted due to fetch error (error=%s)", str(e))
        sys.exit(1)

    if df_raw.empty:
        logger.warning("Ingestion pipeline finished with 0 records fetched")
        return

    # Step 2: Normalize & Impute
    df_normalized = normalize_agmarknet_schema(df_raw)

    # Step 3: Validate
    df_validated = validate_agmarknet_schema(df_normalized)

    # Step 4: Save
    save_raw_agmarknet_data(df_validated, output_dir)

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(
        "Ingestion pipeline completed successfully (records_fetched=%s, records_saved=%s, duration_seconds=%s, output_directory=%s)",
        len(df_raw), len(df_validated), round(duration, 2), output_dir
    )

def main():
    parser = argparse.ArgumentParser(description="Agmarknet Data Ingestion Pipeline for CASPER-Gov")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--commodities", type=str, help="Comma-separated list of commodities to fetch")
    parser.add_argument("--states", type=str, help="Comma-separated list of states to fetch")
    parser.add_argument("--output-dir", type=str, default="data/raw/agmarknet", help="Directory to save Parquet output")

    args = parser.parse_args()
    
    # Process inputs
    commodities = [c.strip() for c in args.commodities.split(",")] if args.commodities else DEFAULT_COMMODITIES
    states = [s.strip() for s in args.states.split(",")] if args.states else DEFAULT_STATES

    run_ingest_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        commodities=commodities,
        states=states,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
