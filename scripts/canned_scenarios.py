import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add root folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.chronos_forecaster import forecast_price_trajectories, compute_projected_breach_risk
from src.rag.vector_store import query_precedents
from src.llm.report_generator import generate_enforcement_notice
from src.audit.logger import log_audit_event, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("canned_scenarios")

def load_models():
    models = {}
    if os.path.exists("models/mapie_conformal.joblib"):
        models["conformal"] = joblib.load("models/mapie_conformal.joblib")
    for name in ["p10", "p50", "p90"]:
        if os.path.exists(f"models/lgb_{name}.joblib"):
            models[name] = joblib.load(f"models/lgb_{name}.joblib")
    return models

def run_scenario_a(models):
    logger.info("--- Triggering Scenario A: Sudden Fuel Price Spike ---")
    
    sku = "Potato"
    state = "Maharashtra"
    
    if not os.path.exists("data/features/test_features.parquet"):
        logger.error("test_features.parquet missing. Cannot execute Scenario A.")
        return None
        
    df = pd.read_parquet("data/features/test_features.parquet")
    df_ts = df[(df["sku_name"] == sku) & (df["state"] == state)].sort_values(by="observation_date")
    
    if df_ts.empty:
        logger.error("No Potato/Maharashtra data found.")
        return None
        
    latest_row = df_ts.iloc[[-1]].copy()
    
    # Simulate a +50% fuel price/freight shock
    fuel_shock_pct = 50.0
    latest_row["macro_pca_3"] += (fuel_shock_pct / 100.0) * 0.5
    
    # Predict bands
    feature_cols = [
        "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
        "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
        "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
        "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
    ]
    cat_cols = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]
    
    if os.path.exists("data/features/commodity_clusters.parquet"):
        df_clusters = pd.read_parquet("data/features/commodity_clusters.parquet")
        latest_row = pd.merge(latest_row, df_clusters[["sku_name", "cluster_id"]], on="sku_name", how="left")
    if "cluster_id" not in latest_row.columns:
        latest_row["cluster_id"] = -1
        
    X = latest_row[feature_cols].copy()
        
    if "conformal" in models:
        for col in cat_cols:
            X[col] = X[col].fillna("Missing").astype(str)
        for col in [f for f in feature_cols if f not in cat_cols]:
            X[col] = X[col].astype(float)
        y_pred, y_pis = models["conformal"].predict_interval(X)
        p10, p50, p90 = float(y_pis[0, 0, 0]), float(y_pred[0]), float(y_pis[0, 1, 0])
    else:
        for col in cat_cols:
            X[col] = X[col].fillna("Missing").astype(str).astype("category")
        p10 = float(models["p10"].predict(X)[0])
        p50 = float(models["p50"].predict(X)[0])
        p90 = float(models["p90"].predict(X)[0])
        
    # Chronos 28-day ahead forecast
    hist_prices = df_ts["modal_price_per_quintal"].tolist()
    med_f, p10_f, p90_f, raw_samples = forecast_price_trajectories(sku, hist_prices, 28)
    
    # Apply scenario shift to forecast
    macro_shift = (fuel_shock_pct / 100.0) * 150.0
    raw_samples = np.maximum(raw_samples + macro_shift, 100.0)
    
    breach_risk = compute_projected_breach_risk(raw_samples, p90)
    
    # Audit log entry
    log_audit_event(
        sku_id=sku,
        region=state,
        model_version="conformal_v1.0" if "conformal" in models else "lgb_quantiles_v1.0",
        feature_snapshot_hash="hash_scenario_a_fuel",
        observed_price=float(latest_row["modal_price_per_quintal"].iloc[0]),
        computed_band={"p10": p10, "p50": p50, "p90": p90},
        anomaly_type="MACRO_SHOCK_RISK",
        llm_verdict_json={"projected_breach_risk_pct": breach_risk, "fuel_price_shock_pct": fuel_shock_pct}
    )
    
    logger.info("Scenario A Completed. Computed ceiling: %s, Breach Risk: %s%%", round(p90, 2), round(breach_risk, 2))
    return {
        "sku": sku,
        "region": state,
        "simulated_fuel_shock_pct": fuel_shock_pct,
        "computed_band": {"p10": p10, "p50": p50, "p90": p90},
        "projected_breach_risk_pct": breach_risk
    }

def run_scenario_b(models):
    logger.info("--- Triggering Scenario B: Essential Medicine Supply Deficit ---")
    
    sku = "Paracetamol"
    state = "Delhi"
    
    # Mock feature values for Paracetamol in Delhi
    mock_row = pd.DataFrame([{
        "price_lag_7d": 1200.0, "price_lag_14d": 1200.0, "price_lag_30d": 1200.0, "price_lag_90d": 1200.0,
        "volatility_7d": 0.02, "volatility_30d": 0.03, "seasonal_index": 1.0, "supply_shock_zscore": -2.0, "is_harvest_season": 0.0,
        "macro_pca_1": 0.0, "macro_pca_2": 0.0, "macro_pca_3": 0.0, "macro_pca_4": 0.0, "macro_pca_5": 0.0,
        "sku_name": sku, "state": state, "district": "New Delhi", "market_mandi": "Delhi Mandi", "sku_variety": "IP_Grade", "cluster_id": "-1"
    }])
    
    # Predict bands
    feature_cols = [
        "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
        "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
        "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
        "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
    ]
    cat_cols = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]
    
    X = mock_row[feature_cols].copy()
    
    if "conformal" in models:
        for col in cat_cols:
            X[col] = X[col].fillna("Missing").astype(str)
        for col in [f for f in feature_cols if f not in cat_cols]:
            X[col] = X[col].astype(float)
        y_pred, y_pis = models["conformal"].predict_interval(X)
        p10, p50, p90 = float(y_pis[0, 0, 0]), float(y_pred[0]), float(y_pis[0, 1, 0])
    elif "p90" in models:
        for col in cat_cols:
            X[col] = X[col].fillna("Missing").astype(str).astype("category")
        p10 = float(models["p10"].predict(X)[0])
        p50 = float(models["p50"].predict(X)[0])
        p90 = float(models["p90"].predict(X)[0])
    else:
        p10, p50, p90 = 1100.0, 1200.0, 1300.0
        
    observed_price = 1500.0
    anomaly_alert = {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "sku_name": sku,
        "state": state,
        "anomaly_type": "PRICE_GOUGING_ALERT",
        "severity_score": 0.82,
        "vendors_involved": ["VEND_PHARMA_09"]
    }
    
    retrieved_precedents = query_precedents("Essential Commodities Act Section 3 pricing control", n_results=1)
    
    drivers = [
        {"feature": "Mandi Arrival Supply Shock", "contribution_percentage": 68.2, "impact_direction": "INCREASE"},
        {"feature": "Recent 7-Day Price Lag", "contribution_percentage": 22.4, "impact_direction": "INCREASE"}
    ]
    notice = generate_enforcement_notice(anomaly_alert, drivers, retrieved_precedents)
    
    # Audit log entry
    log_audit_event(
        sku_id=sku,
        region=state,
        model_version="conformal_v1.0" if "conformal" in models else "lgb_quantiles_v1.0",
        feature_snapshot_hash="hash_scenario_b_med",
        observed_price=observed_price,
        computed_band={"p10": p10, "p50": p50, "p90": p90},
        anomaly_type="PRICE_GOUGING_ALERT",
        llm_verdict_json=json.loads(notice.model_dump_json())
    )
    
    logger.info("Scenario B Completed. Severity: %s, Recommended action: %s", notice.severity_rating, notice.recommended_action)
    return {
        "sku": sku,
        "region": state,
        "observed_price": observed_price,
        "computed_band": {"p10": p10, "p50": p50, "p90": p90},
        "enforcement_notice": json.loads(notice.model_dump_json())
    }

def run_scenario_c():
    logger.info("--- Triggering Scenario C: Regional Retail Cartel Price Synchronization ---")
    
    dates = pd.date_range(start="2026-08-01", periods=10)
    records = []
    
    for day in range(10):
        dt = dates[day]
        for v_idx in range(1, 5):
            vendor_id = f"VEND_ONION_{v_idx}"
            if day == 8 and v_idx in [1, 2, 3]:
                price = 165.0
            else:
                price = 100.0 + np.random.normal(0, 0.4)
                
            records.append({
                "observation_date": dt,
                "vendor_id": vendor_id,
                "state": "Maharashtra",
                "sku_name": "Onion",
                "modal_price_per_quintal": price,
                "macro_pca_1": 0.0
            })
            
    df = pd.DataFrame(records)
    
    # Cartel Behavior detection
    anomaly_objects = []
    for (state, sku), group in df.groupby(["state", "sku_name"]):
        group = group.sort_values(by="observation_date").copy()
        
        vendor_data = {}
        for vend, v_group in group.groupby("vendor_id"):
            v_group = v_group.sort_values(by="observation_date").copy()
            v_group["daily_diff"] = v_group["modal_price_per_quintal"].diff().fillna(0.0)
            v_group["rolling_std"] = v_group["daily_diff"].rolling(7, min_periods=1).std().fillna(0.0)
            v_group["is_spike"] = (v_group["daily_diff"] > 0) & (v_group["daily_diff"] > 2.5 * v_group["rolling_std"])
            vendor_data[vend] = v_group.set_index("observation_date")
            
        unique_dates = sorted(group["observation_date"].unique())
        
        for dt in unique_dates:
            spiked_vendors = []
            for vend, v_df in vendor_data.items():
                if dt in v_df.index:
                    row = v_df.loc[dt]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    if row["is_spike"]:
                        spiked_vendors.append(vend)
                        
            if len(spiked_vendors) >= 3:
                severity = len(spiked_vendors) / len(vendor_data)
                anomaly_objects.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "sku_name": sku,
                    "state": state,
                    "anomaly_type": "CARTEL_BEHAVIOR_FLAG",
                    "severity_score": severity,
                    "vendors_involved": spiked_vendors,
                    "description": f"Cartel price synchronization detected among {len(spiked_vendors)} vendors."
                })
                
    if len(anomaly_objects) > 0:
        log_audit_event(
            sku_id="Onion",
            region="Maharashtra",
            model_version="rule_based_cartel_v1.0",
            feature_snapshot_hash="hash_scenario_c_cartel",
            observed_price=165.0,
            computed_band={"p10": 95.0, "p50": 100.0, "p90": 105.0},
            anomaly_type="CARTEL_BEHAVIOR_FLAG",
            llm_verdict_json=anomaly_objects[0]
        )
        logger.info("Scenario C Completed. Cartel behavior successfully flagged on %s for vendors: %s",
            anomaly_objects[0]["date"], anomaly_objects[0]["vendors_involved"]
        )
        return anomaly_objects[0]
    else:
        logger.warning("Scenario C failed to trigger.")
        return None

def main():
    logger.info("Initializing persistent SQLite Database...")
    init_db("data/audit_log.db")
    
    models = load_models()
    
    output = {}
    
    # 1. Run Scenario A
    res_a = run_scenario_a(models)
    if res_a:
        output["scenario_a"] = res_a
        
    # 2. Run Scenario B
    res_b = run_scenario_b(models)
    if res_b:
        output["scenario_b"] = res_b
        
    # 3. Run Scenario C
    res_c = run_scenario_c()
    if res_c:
        output["scenario_c"] = res_c
        
    out_path = "data/simulations/canned_scenarios_output.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
        
    logger.info("All canned demo scenarios executed and logged successfully to %s", out_path)

if __name__ == "__main__":
    main()
