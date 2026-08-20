import os
import sys
import argparse
import logging
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("macro_pca")

def fit_save_macro_pca(macro_dir: str = "data/raw/macro", ref_dir: str = "data/raw/reference", model_path: str = "models/pca_macro.joblib") -> None:
    logger.info("Starting Macro PCA Fitting (macro_dir=%s, ref_dir=%s)", macro_dir, ref_dir)
    
    # Check paths
    cpi_path = os.path.join(macro_dir, "cpi.parquet")
    wpi_path = os.path.join(macro_dir, "wpi.parquet")
    freight_path = os.path.join(macro_dir, "freight.parquet")
    bench_path = os.path.join(ref_dir, "international_benchmarks.parquet")
    
    for p in [cpi_path, wpi_path, freight_path, bench_path]:
        if not os.path.exists(p):
            logger.error("Required macro/reference file %s does not exist.", p)
            sys.exit(1)
            
    cpi = pd.read_parquet(cpi_path)
    wpi = pd.read_parquet(wpi_path)
    freight = pd.read_parquet(freight_path)
    bench = pd.read_parquet(bench_path)
    
    # Standardize dates
    cpi["observation_date"] = pd.to_datetime(cpi["observation_date"])
    wpi["observation_date"] = pd.to_datetime(wpi["observation_date"])
    freight["observation_date"] = pd.to_datetime(freight["observation_date"])
    bench["observation_date"] = pd.to_datetime(bench["observation_date"])
    
    # Merge CPI and WPI
    df = pd.merge(cpi, wpi, on="observation_date", how="inner")
    # Merge Freight
    df = pd.merge(df, freight, on="observation_date", how="inner")
    # Merge Benchmarks
    df = pd.merge(df, bench, on="observation_date", how="inner")
    
    if df.empty:
        logger.error("No overlapping dates found across macro datasets.")
        sys.exit(1)
        
    # Macro variables to run PCA on
    macro_cols = ["cpi_value", "wpi_value", "freight_index", "fao_food_price_index", "who_health_indicator", "iea_energy_index"]
    
    # Extract features
    X = df[macro_cols].dropna()
    
    if len(X) < 5:
        logger.warning("Fewer than 5 data points to fit PCA. Setting component count to dataset length.")
        n_comp = len(X)
    else:
        n_comp = 5
        
    logger.info("Scaling features and fitting PCA with %s components", n_comp)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_comp)
    pca.fit(X_scaled)
    
    explained_variance = sum(pca.explained_variance_ratio_) * 100.0
    logger.info("PCA Fit complete. Cumulative Explained Variance for %s components: %s%%", n_comp, round(explained_variance, 2))
    
    # Ensure model save directory exists
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Save scaler and pca model
    joblib.dump({"scaler": scaler, "pca": pca, "feature_names": macro_cols}, model_path)
    logger.info("Saved scaler and PCA models successfully to %s", model_path)

def main():
    parser = argparse.ArgumentParser(description="Macro indicators PCA Fitter for CASPER-Gov")
    parser.add_argument("--macro-dir", type=str, default="data/raw/macro", help="Directory containing CPI/WPI/Freight Parquet files")
    parser.add_argument("--ref-dir", type=str, default="data/raw/reference", help="Directory containing international benchmarks Parquet file")
    parser.add_argument("--model-path", type=str, default="models/pca_macro.joblib", help="Output path for serialized models")
    args = parser.parse_args()
    
    fit_save_macro_pca(args.macro_dir, args.ref_dir, args.model_path)

if __name__ == "__main__":
    main()
