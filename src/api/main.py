import os
import sys

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from typing import List, Dict, Any, Optional, Tuple
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("api_main")

app = FastAPI(
    title="CASPER-Gov Regulatory Intelligence API",
    description="API to monitor mandi arrivals, fair price bands, and compliance alerts.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to CASPER-Gov Regulatory Pricing API.",
        "documentation": "/docs",
        "endpoints": {
            "price_bands": "/api/v1/price-bands?sku_name=Tomato&state=Uttar Pradesh",
            "monitoring": "/api/v1/monitoring",
            "anomalies": "/api/v1/anomalies",
            "risk_analysis": "/api/v1/risk-analysis"
        }
    }

# Paths
MODELS_DIR = "models"
FEATURES_DIR = "data/features"
VENDORS_PATH = "data/raw/vendor_registry.parquet"
CLUSTERS_PATH = "data/features/commodity_clusters.parquet"
TEST_FEAT_PATH = "data/features/test_features.parquet"

# Models and data cache
models = {}
df_clusters = None
vendor_map = {}
state_vendor_map = {}

@app.on_event("startup")
def load_resources():
    global df_clusters, vendor_map, state_vendor_map
    logger.info("Initializing API resources and loading ML models...")
    
    # 1. Load LightGBM Quantile models
    for name in ["p10", "p50", "p90"]:
        model_path = os.path.join(MODELS_DIR, f"lgb_{name}.joblib")
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            logger.info("Loaded LightGBM %s model successfully.", name)
        else:
            logger.warning("LightGBM %s model not found at %s.", name, model_path)
            
    # Load Conformal Stacking MAPIE model
    conformal_path = os.path.join(MODELS_DIR, "mapie_conformal.joblib")
    if os.path.exists(conformal_path):
        models["conformal"] = joblib.load(conformal_path)
        logger.info("Loaded Conformal Stacking MAPIE model successfully.")
    else:
        logger.warning("Conformal Stacking MAPIE model not found at %s.", conformal_path)
            
    # 2. Load commodity clusters
    if os.path.exists(CLUSTERS_PATH):
        df_clusters = pd.read_parquet(CLUSTERS_PATH)
        logger.info("Loaded commodity clusters mapping.")
    else:
        logger.warning("Commodity clusters file not found at %s.", CLUSTERS_PATH)
        
    # 3. Load vendor registry and build lookup mappings
    if os.path.exists(VENDORS_PATH):
        vendors_df = pd.read_parquet(VENDORS_PATH)
        for _, row in vendors_df.iterrows():
            st = row["region"]
            skus = [s.strip() for s in row["registered_skus"].split(",")]
            for sku in skus:
                vendor_map.setdefault((st, sku), []).append(row["vendor_id"])
            state_vendor_map.setdefault(st, []).append(row["vendor_id"])
        logger.info("Loaded vendor registry and initialized state/SKU mappings.")
    else:
        logger.warning("Vendor registry file not found at %s.", VENDORS_PATH)

def assign_vendor(state: str, sku: str) -> str:
    """
    Deterministically maps a vendor ID in the same state/region that handles the commodity.
    """
    v_list = vendor_map.get((state, sku), [])
    if not v_list:
        v_list = state_vendor_map.get(state, [])
    if not v_list:
        return "VEND_DEFAULT"
    # Select first vendor deterministically
    return v_list[0]

def get_predictions(df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts predictor columns, casts categoricals, and returns predicted bounds.
    Uses Conformal Stacking MAPIE model if available, otherwise falls back to Quantile LightGBM models.
    """
    feature_cols = [
        "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
        "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
        "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
        "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
    ]
    cat_cols = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]
    
    if "conformal" in models:
        logger.info("Performing inference using Conformal Stacking MAPIE model...")
        X = df_features[feature_cols].copy()
        for col in cat_cols:
            X[col] = X[col].fillna("Missing").astype(str)
        # Ensure numericals are float
        num_cols = [f for f in feature_cols if f not in cat_cols]
        for col in num_cols:
            X[col] = X[col].astype(float)
            
        y_pred, y_pis = models["conformal"].predict_interval(X)
        y_p10 = y_pis[:, 0, 0]
        y_p50 = y_pred
        y_p90 = y_pis[:, 1, 0]
        return y_p10, y_p50, y_p90
        
    logger.info("Performing inference using Quantile LightGBM models...")
    X = df_features[feature_cols].copy()
    for col in cat_cols:
        X[col] = X[col].fillna("Missing").astype(str).astype("category")
        
    y_p10 = models["p10"].predict(X)
    y_p50 = models["p50"].predict(X)
    y_p90 = models["p90"].predict(X)
    
    return y_p10, y_p50, y_p90

@app.get("/api/v1/price-bands")
def get_price_bands(sku_name: str = Query(..., description="SKU/Commodity name"), state: str = Query(..., description="State name")):
    """
    Returns computed price bounds (p10, p50, p90) and observed market price for given SKU and state.
    """
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded. Train models first.")
        
    if not os.path.exists(TEST_FEAT_PATH):
        raise HTTPException(status_code=500, detail="Test features dataset not found.")
        
    df = pd.read_parquet(TEST_FEAT_PATH)
    
    # Filter
    df_filtered = df[(df["sku_name"].str.lower() == sku_name.lower()) & (df["state"].str.lower() == state.lower())].copy()
    if df_filtered.empty:
        raise HTTPException(status_code=404, detail=f"No observations found for SKU '{sku_name}' in region '{state}'.")
        
    # Get latest date's records to represent active pricing
    latest_date = df_filtered["observation_date"].max()
    df_latest = df_filtered[df_filtered["observation_date"] == latest_date].copy()
    
    # Merge clusters mapping
    if df_clusters is not None:
        df_latest = pd.merge(df_latest, df_clusters[["sku_name", "cluster_id"]], on="sku_name", how="left")
    else:
        df_latest["cluster_id"] = -1
        
    # Predict
    try:
        y_p10, y_p50, y_p90 = get_predictions(df_latest)
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    # Build response list for mandis in that state
    markets = []
    for idx, row in df_latest.reset_index(drop=True).iterrows():
        p10 = float(round(y_p10[idx], 2))
        p50 = float(round(y_p50[idx], 2))
        p90 = float(round(y_p90[idx], 2))
        observed = float(round(row["modal_price_per_quintal"], 2))
        
        status = "CEILING_BREACHED" if observed > p90 else "WITHIN_BAND"
        
        markets.append({
            "market_mandi": row["market_mandi"],
            "district": row["district"],
            "sku_variety": row["sku_variety"],
            "observed_price_per_quintal": observed,
            "p10_floor": p10,
            "p50_midpoint": p50,
            "p90_ceiling": p90,
            "compliance_status": status
        })
        
    return {
        "sku_name": sku_name,
        "state": state,
        "latest_observation_date": latest_date.strftime("%Y-%m-%d"),
        "markets": markets
    }

@app.get("/api/v1/monitoring")
def get_monitoring_data():
    """
    Returns full test dataset observations with predicted bands, assigned vendors, and compliance flags.
    """
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded.")
        
    if not os.path.exists(TEST_FEAT_PATH):
        raise HTTPException(status_code=500, detail="Test features dataset not found.")
        
    df = pd.read_parquet(TEST_FEAT_PATH).copy()
    
    # Sort and pick latest date to represent live monitoring view
    latest_date = df["observation_date"].max()
    df_live = df[df["observation_date"] == latest_date].copy()
    
    # Merge cluster ID mappings
    if df_clusters is not None:
        df_live = pd.merge(df_live, df_clusters[["sku_name", "cluster_id"]], on="sku_name", how="left")
    else:
        df_live["cluster_id"] = -1
        
    # Predict
    try:
        y_p10, y_p50, y_p90 = get_predictions(df_live)
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    records = []
    for idx, row in df_live.reset_index(drop=True).iterrows():
        p10 = float(round(y_p10[idx], 2))
        p50 = float(round(y_p50[idx], 2))
        p90 = float(round(y_p90[idx], 2))
        observed = float(round(row["modal_price_per_quintal"], 2))
        
        status = "CEILING_BREACHED" if observed > p90 else "WITHIN_BAND"
        vendor_id = assign_vendor(row["state"], row["sku_name"])
        
        records.append({
            "observation_date": row["observation_date"].strftime("%Y-%m-%d"),
            "sku_name": row["sku_name"],
            "state": row["state"],
            "market_mandi": row["market_mandi"],
            "vendor_id": vendor_id,
            "observed_price": observed,
            "p10_floor": p10,
            "p50_mid": p50,
            "p90_ceiling": p90,
            "compliance_status": status
        })
        
    return records

@app.post("/api/v1/risk-analysis")
async def analyze_batch_risk(file: UploadFile = File(...)):
    """
    Receives an uploaded CSV file, runs model price prediction, and maps risk classification ratings.
    """
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded.")
        
    try:
        content = await file.read()
        df_upload = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {str(e)}")
        
    required_cols = ["sku_name", "state", "district", "market_mandi", "sku_variety", "observed_price"]
    for col in required_cols:
        if col not in df_upload.columns:
            raise HTTPException(status_code=400, detail=f"Missing required CSV column: '{col}'")
            
    # Load latest snapshot to map features
    if os.path.exists(TEST_FEAT_PATH):
        df_feats = pd.read_parquet(TEST_FEAT_PATH)
        df_latest_feats = df_feats.sort_values(by="observation_date").groupby(
            ["sku_name", "state", "district", "market_mandi"]
        ).last().reset_index()
    else:
        df_latest_feats = pd.DataFrame()
        
    analyzed_records = []
    
    # Process row by row
    rows_to_predict = []
    for idx, row in df_upload.iterrows():
        # Match latest features
        match = pd.DataFrame()
        if not df_latest_feats.empty:
            match = df_latest_feats[
                (df_latest_feats["sku_name"].str.lower() == str(row["sku_name"]).lower()) &
                (df_latest_feats["state"].str.lower() == str(row["state"]).lower()) &
                (df_latest_feats["district"].str.lower() == str(row["district"]).lower()) &
                (df_latest_feats["market_mandi"].str.lower() == str(row["market_mandi"]).lower())
            ]
            
        # Copy features
        feat_dict = {
            "sku_name": row["sku_name"],
            "state": row["state"],
            "district": row["district"],
            "market_mandi": row["market_mandi"],
            "sku_variety": row["sku_variety"],
            "modal_price_per_quintal": float(row["observed_price"])
        }
        
        if not match.empty:
            match_row = match.iloc[0]
            for col in [
                "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
                "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
                "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5"
            ]:
                feat_dict[col] = match_row[col]
        else:
            # Fallback defaults
            op = float(row["observed_price"])
            feat_dict.update({
                "price_lag_7d": op, "price_lag_14d": op, "price_lag_30d": op, "price_lag_90d": op,
                "volatility_7d": 0.05, "volatility_30d": 0.05, "seasonal_index": 1.0, "supply_shock_zscore": 0.0, "is_harvest_season": 0.0,
                "macro_pca_1": 0.0, "macro_pca_2": 0.0, "macro_pca_3": 0.0, "macro_pca_4": 0.0, "macro_pca_5": 0.0
            })
            
        rows_to_predict.append(feat_dict)
        
    df_batch = pd.DataFrame(rows_to_predict)
    
    # Merge clusters
    if df_clusters is not None:
        df_batch = pd.merge(df_batch, df_clusters[["sku_name", "cluster_id"]], on="sku_name", how="left")
    if "cluster_id" not in df_batch.columns:
        df_batch["cluster_id"] = -1
        
    # Run predictions
    try:
        y_p10, y_p50, y_p90 = get_predictions(df_batch)
    except Exception as e:
        logger.error("Batch inference failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")
        
    # Construct response
    for idx, row in df_upload.iterrows():
        p10 = float(round(y_p10[idx], 2))
        p50 = float(round(y_p50[idx], 2))
        p90 = float(round(y_p90[idx], 2))
        observed = float(row["observed_price"])
        
        # Risk classification
        if observed > p90:
            risk = "HIGH RISK"
        elif observed > p50:
            risk = "MEDIUM RISK"
        else:
            risk = "LOW RISK"
            
        analyzed_records.append({
            "sku_name": row["sku_name"],
            "state": row["state"],
            "district": row["district"],
            "market_mandi": row["market_mandi"],
            "sku_variety": row["sku_variety"],
            "observed_price": observed,
            "p10_floor": p10,
            "p50_midpoint": p50,
            "p90_ceiling": p90,
            "risk_rating": risk
        })
        
    return analyzed_records


@app.get("/api/v1/anomalies")
def get_anomalies(
    sku_name: Optional[str] = Query(None, description="Filter anomalies by SKU/Commodity name"),
    state: Optional[str] = Query(None, description="Filter anomalies by state/region"),
    anomaly_type: Optional[str] = Query(None, description="Filter by PRICE_GOUGING or CARTEL_SPIKE"),
    contamination: float = Query(0.05, ge=0.01, le=0.5, description="Contamination factor for IsolationForest"),
    cartel_std_threshold: float = Query(2.5, ge=1.0, le=5.0, description="Z-score threshold for cartel spikes"),
    min_vendors: int = Query(3, ge=2, le=10, description="Minimum synchronized vendors for cartel detection")
):
    """
    Detects price gouging and cartel anomalies across active mandi observations.
    Leverages the CASPER-Gov AnomalyDetector engine.
    """
    if not os.path.exists(TEST_FEAT_PATH):
        raise HTTPException(status_code=500, detail="Test features dataset not found.")

    try:
        from src.models.anomaly_detector import AnomalyDetector
        
        df = pd.read_parquet(TEST_FEAT_PATH)
        if df.empty:
            return []

        detector = AnomalyDetector(contamination=contamination)
        
        # 1. Price gouging detection
        df_gouging = detector.detect_price_gouging(df)
        
        # 2. Cartel spike detection
        cartel_alerts = detector.detect_cartel_spikes(
            df, 
            std_threshold=cartel_std_threshold, 
            min_vendors=min_vendors
        )
        
        results = []
        
        # Collect gouging anomalies
        gouging_rows = df_gouging[df_gouging["is_gouging"]].copy()
        for _, row in gouging_rows.iterrows():
            results.append({
                "observation_id": str(row.get("id", f"{row['observation_date']}_{row['sku_name']}_{row['market_mandi']}")),
                "date": pd.Timestamp(row["observation_date"]).strftime("%Y-%m-%d"),
                "sku_name": row["sku_name"],
                "state": row.get("state", ""),
                "market_mandi": row["market_mandi"],
                "observed_price": float(row["modal_price_per_quintal"]),
                "anomaly_type": "PRICE_GOUGING",
                "severity_score": round(float(row["gouging_severity"]), 4),
                "details": {
                    "anomaly_raw_score": round(float(row["anomaly_raw_score"]), 6),
                }
            })
            
        # Collect cartel spike anomalies
        for alert in cartel_alerts:
            results.append({
                "observation_id": alert.observation_id,
                "date": str(alert.details.get("observation_date", ""))[:10],
                "sku_name": alert.sku_name,
                "state": alert.region,
                "market_mandi": alert.vendor_or_mandi,
                "observed_price": alert.observed_price,
                "anomaly_type": alert.anomaly_type,
                "severity_score": round(alert.severity_score, 4),
                "details": alert.details
            })

        # Apply optional query filters
        if sku_name:
            results = [r for r in results if r["sku_name"].lower() == sku_name.lower()]
        if state:
            results = [r for r in results if r["state"].lower() == state.lower()]
        if anomaly_type:
            results = [r for r in results if r["anomaly_type"].lower() == anomaly_type.lower()]

        return results
    except Exception as e:
        logger.error("Anomaly detection endpoint error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

