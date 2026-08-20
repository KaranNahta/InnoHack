import os
import sys
import argparse
import logging
import pandas as pd
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("build_features")

def transform_features(df: pd.DataFrame, macro_dir: str = "data/raw/macro", ref_dir: str = "data/raw/reference", model_path: str = "models/pca_macro.joblib") -> pd.DataFrame:
    """
    Transforms raw/processed mandi data by adding rolling features, lags, volatilities,
    seasonal indexes, supply shock z-scores, harvest flags, and PCA-reduced macro components.
    """
    logger.info("Transforming features for mandi dataframe (records=%d)...", len(df))
    
    if df.empty:
        return df

    # Create copy to avoid mutating original
    df = df.copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values("observation_date")

    # 1. Calculate regional price daily average for lags
    df_reg = df.groupby(["sku_name", "state", "observation_date"])["modal_price_per_quintal"].mean().reset_index()
    df_reg = df_reg.rename(columns={"modal_price_per_quintal": "reg_price"})
    
    # Compute lags (1w = 7d, 2w = 14d, 1m = 30d, 3m = 90d)
    for l in [7, 14, 30, 90]:
        df_shifted = df_reg.copy()
        df_shifted["observation_date"] = df_shifted["observation_date"] + pd.Timedelta(days=l)
        df_shifted = df_shifted.rename(columns={"reg_price": f"price_lag_{l}d"})
        df = pd.merge(df, df_shifted, on=["sku_name", "state", "observation_date"], how="left")
        # Fill missing lags with forward fill per group or default to current price
        df[f"price_lag_{l}d"] = df.groupby(["sku_name", "state", "market_mandi"])[f"price_lag_{l}d"].ffill()
        df[f"price_lag_{l}d"] = df[f"price_lag_{l}d"].fillna(df["modal_price_per_quintal"])

    # 2. Compute rolling price volatility (7-day and 30-day standard deviation)
    # Set index to observation_date for rolling calculations
    df_idx = df.set_index("observation_date").sort_index()
    
    # Rolling standard deviation per group
    vol_7d_series = df_idx.groupby(["sku_name", "state", "market_mandi"])["modal_price_per_quintal"].rolling(window="7D").std().reset_index()
    vol_7d_series = vol_7d_series.rename(columns={"modal_price_per_quintal": "volatility_7d"})
    
    vol_30d_series = df_idx.groupby(["sku_name", "state", "market_mandi"])["modal_price_per_quintal"].rolling(window="30D").std().reset_index()
    vol_30d_series = vol_30d_series.rename(columns={"modal_price_per_quintal": "volatility_30d"})
    
    # Merge volatilities back to main df
    df = pd.merge(df, vol_7d_series, on=["sku_name", "state", "market_mandi", "observation_date"], how="left")
    df = pd.merge(df, vol_30d_series, on=["sku_name", "state", "market_mandi", "observation_date"], how="left")
    df["volatility_7d"] = df["volatility_7d"].fillna(0.0)
    df["volatility_30d"] = df["volatility_30d"].fillna(0.0)

    # 3. Compute Seasonal Index
    month = df["observation_date"].dt.month
    monthly_mean = df.groupby(["sku_name", month])["modal_price_per_quintal"].transform("mean")
    overall_mean = df.groupby(["sku_name"])["modal_price_per_quintal"].transform("mean")
    df["seasonal_index"] = (monthly_mean / (overall_mean + 1e-5)).fillna(1.0)

    # 4. Compute Supply Shock Z-Score
    mean_arr_series = df_idx.groupby(["sku_name", "state", "market_mandi"])["arrival_quantity_tonnes"].rolling(window="30D").mean().reset_index()
    mean_arr_series = mean_arr_series.rename(columns={"arrival_quantity_tonnes": "rolling_mean_arr"})
    
    std_arr_series = df_idx.groupby(["sku_name", "state", "market_mandi"])["arrival_quantity_tonnes"].rolling(window="30D").std().reset_index()
    std_arr_series = std_arr_series.rename(columns={"arrival_quantity_tonnes": "rolling_std_arr"})
    
    # Merge and calculate z-score
    df = pd.merge(df, mean_arr_series, on=["sku_name", "state", "market_mandi", "observation_date"], how="left")
    df = pd.merge(df, std_arr_series, on=["sku_name", "state", "market_mandi", "observation_date"], how="left")
    
    df["supply_shock_zscore"] = ((df["arrival_quantity_tonnes"] - df["rolling_mean_arr"]) / (df["rolling_std_arr"] + 1e-5)).fillna(0.0)
    df = df.drop(columns=["rolling_mean_arr", "rolling_std_arr"])

    # 5. Harvest Cycle Flags
    harvest_path = os.path.join(ref_dir, "harvest.parquet")
    if not os.path.exists(harvest_path):
        harvest_path = os.path.join(macro_dir, "harvest.parquet")
        
    if os.path.exists(harvest_path):
        harvest_df = pd.read_parquet(harvest_path)
        df = pd.merge(df, harvest_df, on=["sku_name", "state"], how="left")
        
        # Calculate flag
        m_obs = df["observation_date"].dt.month
        start = df["harvest_start_month"]
        end = df["harvest_end_month"]
        
        is_harvest = (
            ((start <= end) & (m_obs >= start) & (m_obs <= end)) |
            ((start > end) & ((m_obs >= start) | (m_obs <= end)))
        ).astype(float).fillna(0.0)
        df["is_harvest_season"] = is_harvest
        df = df.drop(columns=["harvest_start_month", "harvest_end_month", "harvest_season"], errors="ignore")
    else:
        logger.warning("Harvest reference parquet not found. Setting is_harvest_season to 0.")
        df["is_harvest_season"] = 0.0

    # 6. Macro PCA components
    cpi_path = os.path.join(macro_dir, "cpi.parquet")
    wpi_path = os.path.join(macro_dir, "wpi.parquet")
    freight_path = os.path.join(macro_dir, "freight.parquet")
    bench_path = os.path.join(ref_dir, "international_benchmarks.parquet")
    
    if all(os.path.exists(p) for p in [cpi_path, wpi_path, freight_path, bench_path]):
        cpi = pd.read_parquet(cpi_path)
        wpi = pd.read_parquet(wpi_path)
        freight = pd.read_parquet(freight_path)
        bench = pd.read_parquet(bench_path)
        
        cpi["observation_date"] = pd.to_datetime(cpi["observation_date"])
        wpi["observation_date"] = pd.to_datetime(wpi["observation_date"])
        freight["observation_date"] = pd.to_datetime(freight["observation_date"])
        bench["observation_date"] = pd.to_datetime(bench["observation_date"])
        
        df_macro = pd.merge(cpi, wpi, on="observation_date", how="inner")
        df_macro = pd.merge(df_macro, freight, on="observation_date", how="inner")
        df_macro = pd.merge(df_macro, bench, on="observation_date", how="inner")
        
        df_macro["year_month"] = df_macro["observation_date"].dt.to_period("M")
        df["year_month"] = df["observation_date"].dt.to_period("M")
        
        macro_cols = ["cpi_value", "wpi_value", "freight_index", "fao_food_price_index", "who_health_indicator", "iea_energy_index"]
        df = pd.merge(df, df_macro[["year_month"] + macro_cols], on="year_month", how="left")
        
        # Load PCA model
        if not os.path.exists(model_path):
            logger.warning("Fitted PCA model not found at %s. Triggering fitting process...", model_path)
            from src.features.macro_pca import fit_save_macro_pca
            fit_save_macro_pca(macro_dir, ref_dir, model_path)
            
        pca_data = joblib.load(model_path)
        scaler = pca_data["scaler"]
        pca = pca_data["pca"]
        
        # Fill any missing macro variables with forward/backward fills
        df[macro_cols] = df[macro_cols].ffill().bfill().fillna(0.0)
        
        scaled_macro = scaler.transform(df[macro_cols])
        pca_features = pca.transform(scaled_macro)
        
        for i in range(pca_features.shape[1]):
            df[f"macro_pca_{i+1}"] = pca_features[:, i]
            
        # Drop temporary keys and raw macro columns
        df = df.drop(columns=macro_cols + ["year_month"], errors="ignore")
    else:
        logger.warning("Macro data files not found. Setting macro PCA components to 0.")
        for i in range(1, 6):
            df[f"macro_pca_{i}"] = 0.0

    return df

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline for CASPER-Gov")
    parser.add_argument("--input-dir", type=str, default="data/processed", help="Directory containing train/val/test splits")
    parser.add_argument("--output-dir", type=str, default="data/features", help="Directory to save engineered features")
    parser.add_argument("--macro-dir", type=str, default="data/raw/macro", help="Directory containing CPI/WPI/Freight Parquet files")
    parser.add_argument("--ref-dir", type=str, default="data/raw/reference", help="Directory containing international benchmarks and harvest calendar")
    parser.add_argument("--model-path", type=str, default="models/pca_macro.joblib", help="Path to PCA model file")
    
    args = parser.parse_args()
    
    logger.info("Starting feature engineering pipeline...")
    
    # Check if inputs exist
    for f_name in ["train.parquet", "val.parquet", "test.parquet"]:
        p = os.path.join(args.input_dir, f_name)
        if not os.path.exists(p):
            logger.error("Required split dataset %s does not exist.", p)
            sys.exit(1)
            
    # Load splits
    df_train = pd.read_parquet(os.path.join(args.input_dir, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(args.input_dir, "val.parquet"))
    df_test = pd.read_parquet(os.path.join(args.input_dir, "test.parquet"))
    
    # Process
    df_train_feat = transform_features(df_train, args.macro_dir, args.ref_dir, args.model_path)
    df_val_feat = transform_features(df_val, args.macro_dir, args.ref_dir, args.model_path)
    df_test_feat = transform_features(df_test, args.macro_dir, args.ref_dir, args.model_path)
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    df_train_feat.to_parquet(os.path.join(args.output_dir, "train_features.parquet"), compression="snappy", index=False)
    df_val_feat.to_parquet(os.path.join(args.output_dir, "val_features.parquet"), compression="snappy", index=False)
    df_test_feat.to_parquet(os.path.join(args.output_dir, "test_features.parquet"), compression="snappy", index=False)
    
    logger.info("Feature engineering pipeline completed successfully.")

if __name__ == "__main__":
    main()
