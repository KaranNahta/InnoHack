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
logger = logging.getLogger("temporal_split")

def perform_temporal_split(input_dir: str, output_dir: str) -> None:
    """
    Performs chronological temporal split (60% Train, 20% Val, 20% Test) on the mandi pricing data.
    Enforces point-in-time constraints to prevent target leakage.
    """
    logger.info("Starting temporal splitting process (input_dir=%s, output_dir=%s)", input_dir, output_dir)
    
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        logger.error("Input directory %s does not exist or is empty.", input_dir)
        sys.exit(1)
        
    try:
        # Load all partitioned raw parquet data
        df = pd.read_parquet(input_dir)
    except Exception as e:
        logger.error("Failed to read raw parquet data from %s (error=%s)", input_dir, str(e))
        raise e
        
    if df.empty:
        logger.warning("Empty DataFrame loaded from %s. Cannot perform temporal split.", input_dir)
        return

    # Ensure observation_date is datetime64[ns]
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    
    # Get unique dates and sort chronologically
    unique_dates = sorted(df["observation_date"].unique())
    n_dates = len(unique_dates)
    
    if n_dates < 3:
        logger.error("Too few unique dates (%d) to perform a 60/20/20 split.", n_dates)
        sys.exit(1)
        
    # Calculate cutoff indices based on unique dates
    train_end_idx = int(0.6 * n_dates)
    val_end_idx = int(0.8 * n_dates)
    
    # Partition unique dates
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    # Split the DataFrame
    df_train = df[df["observation_date"].isin(train_dates)].copy()
    df_val = df[df["observation_date"].isin(val_dates)].copy()
    df_test = df[df["observation_date"].isin(test_dates)].copy()
    
    total_records = len(df)
    
    # Statistical verification logs
    logger.info("=== Temporal Split Verification ===")
    
    splits = [("Train", df_train, train_dates), ("Val", df_val, val_dates), ("Test", df_test, test_dates)]
    
    for name, split_df, split_dates in splits:
        count = len(split_df)
        pct = (count / total_records) * 100.0 if total_records > 0 else 0.0
        # Convert pandas timestamp to datetime object for formatted printing
        start_date = pd.Timestamp(split_dates[0]).to_pydatetime().strftime("%Y-%m-%d")
        end_date = pd.Timestamp(split_dates[-1]).to_pydatetime().strftime("%Y-%m-%d")
        logger.info(
            "%s Split: %s records (%s%%) | Date Range: %s to %s | Unique Dates: %s",
            name, count, round(pct, 2), start_date, end_date, len(split_dates)
        )
        
    # Double check date ranges do not overlap
    assert max(train_dates) < min(val_dates), "Train and Validation date ranges overlap!"
    assert max(val_dates) < min(test_dates), "Validation and Test date ranges overlap!"
    logger.info("Validation passed: date ranges are strictly chronological and non-overlapping.")
    
    # Write output Parquet files
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df_train.to_parquet(os.path.join(output_dir, "train.parquet"), compression="snappy", index=False)
        df_val.to_parquet(os.path.join(output_dir, "val.parquet"), compression="snappy", index=False)
        df_test.to_parquet(os.path.join(output_dir, "test.parquet"), compression="snappy", index=False)
        logger.info("Successfully saved train.parquet, val.parquet, and test.parquet in %s", output_dir)
    except Exception as e:
        logger.error("Failed to write split Parquet files (error=%s)", str(e))
        raise e

def main():
    parser = argparse.ArgumentParser(description="Temporal Train/Val/Test Split for CASPER-Gov")
    parser.add_argument("--input-dir", type=str, default="data/raw/agmarknet", help="Directory containing raw agmarknet Parquet data")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save train/val/test splits")
    
    args = parser.parse_args()
    perform_temporal_split(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
