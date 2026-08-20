import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def test_isolation_forest_anomaly_gouging():
    # 1. Create synthetic vendor dataset
    # Normal daily fluctuations: mean 100, std 2
    np.random.seed(42)
    normal_prices = np.random.normal(100.0, 2.0, size=99)
    # Add one extreme price gouging spike of 160 on day 100
    prices = np.append(normal_prices, [160.0])
    
    df = pd.DataFrame({
        "modal_price_per_quintal": prices,
        "macro_pca_1": np.zeros(100) # stable macro
    })
    
    # Compute features for Isolation Forest
    df["price_pct_change"] = df["modal_price_per_quintal"].pct_change().fillna(0.0)
    median_price = df["modal_price_per_quintal"].median()
    df["dev_from_median"] = (df["modal_price_per_quintal"] - median_price) / median_price
    df["price_std_score"] = (df["modal_price_per_quintal"] - df["modal_price_per_quintal"].mean()) / df["modal_price_per_quintal"].std()
    df["price_dev_vs_macro"] = df["price_std_score"] - df["macro_pca_1"]
    
    features = ["price_pct_change", "dev_from_median", "price_dev_vs_macro"]
    X = df[features].copy()
    
    # Fit Isolation Forest
    iso = IsolationForest(contamination=0.02, random_state=42)
    df["is_anomaly"] = iso.fit_predict(X) == -1
    
    # Verify the extreme spike on day 100 is flagged
    assert df.loc[99, "is_anomaly"] == True, "Extreme price gouging spike was not flagged by IsolationForest."
    # Most normal days should not be anomalies
    assert df.loc[1:90, "is_anomaly"].mean() < 0.05

def test_synchronized_cartel_spikes():
    # 2. Test retail cartel price synchronization detection mechanics
    # 4 vendors over 10 days in the same region
    dates = pd.date_range(start="2026-08-01", periods=10)
    records = []
    
    # Vendors 1, 2, 3 spike together on day 9. Vendor 4 stays stable.
    for day in range(10):
        dt = dates[day]
        for v_idx in range(1, 5):
            vendor_id = f"VEND_{v_idx}"
            if day == 8 and v_idx in [1, 2, 3]:
                price = 150.0  # Spiked!
            else:
                price = 100.0 + np.random.normal(0, 0.5)
                
            records.append({
                "observation_date": dt,
                "vendor_id": vendor_id,
                "state": "Maharashtra",
                "sku_name": "Onion",
                "modal_price_per_quintal": price,
                "macro_pca_1": 0.0 # stable macro
            })
            
    df = pd.DataFrame(records)
    
    # Run the same cartel check logic as src/models/anomaly_detector.py
    anomaly_objects = []
    
    for (state, sku), group in df.groupby(["state", "sku_name"]):
        group = group.sort_values(by="observation_date").copy()
        
        vendor_data = {}
        for vend, v_group in group.groupby("vendor_id"):
            v_group = v_group.sort_values(by="observation_date").copy()
            v_group["daily_diff"] = v_group["modal_price_per_quintal"].diff().fillna(0.0)
            v_group["rolling_std"] = v_group["daily_diff"].rolling(7, min_periods=1).std().fillna(0.0)
            # A spike is a positive increase > 2.5 times rolling standard deviation
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
                        
            # Check cartel condition
            if len(spiked_vendors) >= 3:
                severity = len(spiked_vendors) / len(vendor_data)
                anomaly_objects.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "sku_name": sku,
                    "state": state,
                    "anomaly_type": "CARTEL_BEHAVIOR_FLAG",
                    "severity_score": severity,
                    "vendors_involved": spiked_vendors
                })
                
    # Verify that a cartel behavior flag was generated on day 9 (index 8)
    assert len(anomaly_objects) == 1
    assert anomaly_objects[0]["anomaly_type"] == "CARTEL_BEHAVIOR_FLAG"
    assert anomaly_objects[0]["date"] == "2026-08-09"
    assert sorted(anomaly_objects[0]["vendors_involved"]) == ["VEND_1", "VEND_2", "VEND_3"]
