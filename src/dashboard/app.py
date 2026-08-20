import os
import sys

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import io
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.graph_objects as go

from src.models.chronos_forecaster import forecast_price_trajectories, compute_projected_breach_risk

# Page configuration
st.set_page_config(
    page_title="CASPER-Gov Regulatory Intelligence Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h3 {
        font-family: 'Inter', sans-serif;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# API and File Paths
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_MONITORING_URL = f"http://{API_HOST}:8000/api/v1/monitoring"
API_RISK_URL = f"http://{API_HOST}:8000/api/v1/risk-analysis"
TEST_FEAT_PATH = "data/features/test_features.parquet"
VENDORS_PATH = "data/raw/vendor_registry.parquet"
CLUSTERS_PATH = "data/features/commodity_clusters.parquet"

# Models and data cache
@st.cache_resource
def load_ml_resources():
    models = {}
    conformal_path = "models/mapie_conformal.joblib"
    if os.path.exists(conformal_path):
        models["conformal"] = joblib.load(conformal_path)
    for name in ["p10", "p50", "p90"]:
        path = f"models/lgb_{name}.joblib"
        if os.path.exists(path):
            models[name] = joblib.load(path)
            
    df_clusters = None
    if os.path.exists(CLUSTERS_PATH):
        df_clusters = pd.read_parquet(CLUSTERS_PATH)
        
    return models, df_clusters

models, df_clusters = load_ml_resources()

def predict_local_batch(df_up: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback method to run batch predictions locally if FastAPI is offline.
    """
    feature_cols = [
        "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
        "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
        "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
        "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
    ]
    cat_cols = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]
    
    # Load latest snapshot
    if os.path.exists(TEST_FEAT_PATH):
        df_feats = pd.read_parquet(TEST_FEAT_PATH)
        df_latest_feats = df_feats.sort_values(by="observation_date").groupby(
            ["sku_name", "state", "district", "market_mandi"]
        ).last().reset_index()
    else:
        df_latest_feats = pd.DataFrame()
        
    rows_to_predict = []
    for idx, row in df_up.iterrows():
        match = pd.DataFrame()
        if not df_latest_feats.empty:
            match = df_latest_feats[
                (df_latest_feats["sku_name"].str.lower() == str(row["sku_name"]).lower()) &
                (df_latest_feats["state"].str.lower() == str(row["state"]).lower()) &
                (df_latest_feats["district"].str.lower() == str(row["district"]).lower()) &
                (df_latest_feats["market_mandi"].str.lower() == str(row["market_mandi"]).lower())
            ]
            
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
            op = float(row["observed_price"])
            feat_dict.update({
                "price_lag_7d": op, "price_lag_14d": op, "price_lag_30d": op, "price_lag_90d": op,
                "volatility_7d": 0.05, "volatility_30d": 0.05, "seasonal_index": 1.0, "supply_shock_zscore": 0.0, "is_harvest_season": 0.0,
                "macro_pca_1": 0.0, "macro_pca_2": 0.0, "macro_pca_3": 0.0, "macro_pca_4": 0.0, "macro_pca_5": 0.0
            })
            
        rows_to_predict.append(feat_dict)
        
    df_batch = pd.DataFrame(rows_to_predict)
    if df_clusters is not None:
        df_batch = pd.merge(df_batch, df_clusters[["sku_name", "cluster_id"]], on="sku_name", how="left")
    if "cluster_id" not in df_batch.columns:
        df_batch["cluster_id"] = -1
        
    if "conformal" in models:
        for col in cat_cols:
            X = df_batch[feature_cols].copy()
            X[col] = X[col].fillna("Missing").astype(str)
        for col in [f for f in feature_cols if f not in cat_cols]:
            X[col] = X[col].astype(float)
        y_pred, y_pis = models["conformal"].predict_interval(X)
        y_p10 = y_pis[:, 0, 0]
        y_p50 = y_pred
        y_p90 = y_pis[:, 1, 0]
    else:
        X = df_batch[feature_cols].copy()
        for col in cat_cols:
            X[col] = X[col].fillna("Missing").astype(str).astype("category")
        y_p10 = models["p10"].predict(X)
        y_p50 = models["p50"].predict(X)
        y_p90 = models["p90"].predict(X)
        
    records = []
    for idx, row in df_up.iterrows():
        p10 = float(round(y_p10[idx], 2))
        p50 = float(round(y_p50[idx], 2))
        p90 = float(round(y_p90[idx], 2))
        observed = float(row["observed_price"])
        
        if observed > p90:
            risk = "HIGH RISK"
        elif observed > p50:
            risk = "MEDIUM RISK"
        else:
            risk = "LOW RISK"
            
        records.append({
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
    return pd.DataFrame(records)

@st.cache_data(ttl=60)
def load_monitoring_data():
    """
    Fetches live monitoring data from FastAPI.
    Falls back to direct computation if the API is offline.
    """
    try:
        response = requests.get(API_MONITORING_URL, timeout=3)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
    except Exception as e:
        if all(os.path.exists(p) for p in [TEST_FEAT_PATH, VENDORS_PATH, CLUSTERS_PATH]):
            df = pd.read_parquet(TEST_FEAT_PATH)
            latest_date = df["observation_date"].max()
            df_live = df[df["observation_date"] == latest_date].copy()
            
            if df_clusters is not None:
                df_live = pd.merge(df_live, df_clusters[["sku_name", "cluster_id"]], on="sku_name", how="left")
            
            feature_cols = [
                "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
                "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
                "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
                "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
            ]
            cat_cols = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]
            
            X = df_live[feature_cols].copy()
            
            if "conformal" in models:
                for col in cat_cols:
                    X[col] = X[col].fillna("Missing").astype(str)
                num_cols = [f for f in feature_cols if f not in cat_cols]
                for col in num_cols:
                    X[col] = X[col].astype(float)
                y_pred, y_pis = models["conformal"].predict_interval(X)
                y_p10 = y_pis[:, 0, 0]
                y_p50 = y_pred
                y_p90 = y_pis[:, 1, 0]
            else:
                for col in cat_cols:
                    X[col] = X[col].fillna("Missing").astype(str).astype("category")
                y_p10 = models["p10"].predict(X)
                y_p50 = models["p50"].predict(X)
                y_p90 = models["p90"].predict(X)
            
            vendors_df = pd.read_parquet(VENDORS_PATH)
            vendor_map = {}
            for _, row in vendors_df.iterrows():
                skus = [s.strip() for s in row["registered_skus"].split(",")]
                for sku in skus:
                    vendor_map[(row["region"], sku)] = row["vendor_id"]
                    
            records = []
            for idx, row in df_live.reset_index(drop=True).iterrows():
                p10 = float(round(y_p10[idx], 2))
                p50 = float(round(y_p50[idx], 2))
                p90 = float(round(y_p90[idx], 2))
                observed = float(round(row["modal_price_per_quintal"], 2))
                
                status = "CEILING_BREACHED" if observed > p90 else "WITHIN_BAND"
                vendor_id = vendor_map.get((row["state"], row["sku_name"]), "VEND_0001")
                
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
            return pd.DataFrame(records)
            
    return pd.DataFrame()

# Fetch data
df_monitor = load_monitoring_data()

if df_monitor.empty:
    st.error("Error: Could not retrieve monitoring data. Check that feature pipelines have completed and models are trained.")
    st.stop()

# Title
st.title("⚖️ CASPER-Gov Live Regulatory Pricing Monitor")
st.subheader("Price Band Estimation & statutory ceiling compliance alerts")

# Tabs
tab_audit, tab_scenario, tab_upload = st.tabs([
    "📋 Live Audit Monitor", 
    "⚡ Scenario Planning & Risk Forecasting",
    "📤 Batch Risk Uploader"
])

# --- TAB 1: AUDIT MONITOR ---
with tab_audit:
    st.sidebar.header("🔍 Filters Configuration")
    states_list = sorted(list(df_monitor["state"].unique()))
    selected_states = st.sidebar.multiselect("Select Regions/States", states_list, default=states_list)
    sku_list = sorted(list(df_monitor["sku_name"].unique()))
    selected_skus = st.sidebar.multiselect("Select Commodity Category", sku_list, default=sku_list)
    status_list = sorted(list(df_monitor["compliance_status"].unique()))
    selected_status = st.sidebar.multiselect("Select Compliance Status", status_list, default=status_list)

    df_filtered = df_monitor[
        (df_monitor["state"].isin(selected_states)) &
        (df_monitor["sku_name"].isin(selected_skus)) &
        (df_monitor["compliance_status"].isin(selected_status))
    ]

    col1, col2, col3, col4 = st.columns(4)
    total_mandis = len(df_filtered)
    breached_count = len(df_filtered[df_filtered["compliance_status"] == "CEILING_BREACHED"])
    compliance_rate = ((total_mandis - breached_count) / total_mandis * 100.0) if total_mandis > 0 else 100.0
    avg_price = df_filtered["observed_price"].mean() if total_mandis > 0 else 0.0

    with col1:
        st.metric(label="Total Mandis Monitored", value=f"{total_mandis}")
    with col2:
        st.metric(label="Price Ceiling Breaches", value=f"{breached_count}", delta=f"{breached_count} alerts", delta_color="inverse")
    with col3:
        st.metric(label="Compliance Rate", value=f"{compliance_rate:.1f}%")
    with col4:
        st.metric(label="Average Observed Price (INR/Qtl)", value=f"₹{avg_price:.2f}")

    st.write("---")
    st.markdown("### 📋 Active Market Pricing Audit Table")

    def highlight_breaches(row):
        status = row["Compliance Status"]
        if status == "CEILING_BREACHED":
            return ["background-color: #7f1d1d; color: #fecaca; font-weight: bold"] * len(row)
        else:
            return ["background-color: #064e3b; color: #d1fae5"] * len(row)

    df_presentation = df_filtered[[
        "sku_name", "state", "market_mandi", "vendor_id", 
        "observed_price", "p10_floor", "p50_mid", "p90_ceiling", "compliance_status"
    ]].rename(columns={
        "sku_name": "Commodity SKU",
        "state": "Region/State",
        "market_mandi": "Mandi/Market",
        "vendor_id": "Assigned Vendor",
        "observed_price": "Observed Price (INR/Qtl)",
        "p10_floor": "Floor (p10)",
        "p50_mid": "Fair Mid (p50)",
        "p90_ceiling": "Ceiling (p90)",
        "compliance_status": "Compliance Status"
    })

    if not df_presentation.empty:
        rows_per_page = 100
        n_pages = int(np.ceil(len(df_presentation) / rows_per_page))
        
        pag_col1, pag_col2 = st.columns([1, 4])
        with pag_col1:
            page_number = st.number_input("Page", min_value=1, max_value=max(1, n_pages), value=1, step=1, key="audit_page_num")
        with pag_col2:
            start_idx = (page_number - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(df_presentation))
            st.markdown(f"<div style='padding-top: 10px; color: #94a3b8;'>Showing records <b>{start_idx + 1}</b> to <b>{end_idx}</b> of <b>{len(df_presentation)}</b></div>", unsafe_allow_html=True)
            
        df_page = df_presentation.iloc[start_idx:end_idx]
        st.dataframe(
            df_page.style.apply(highlight_breaches, axis=1),
            use_container_width=True,
            height=500
        )
    else:
        st.warning("No records match the current filter configuration.")

# --- TAB 2: SCENARIO PLANNING ---
with tab_scenario:
    st.markdown("### ⚡ Policy Leverage Simulator & Risk Forecaster")
    st.write("Select a SKU and State to model simulated interventions and project price trajectories.")
    
    if not os.path.exists(TEST_FEAT_PATH):
        st.error("Test features parquet dataset not found. Verify processing pipeline.")
    else:
        df_all = pd.read_parquet(TEST_FEAT_PATH)
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            selected_sku = st.selectbox("Select Commodity SKU", sorted(df_all["sku_name"].unique()), key="scenario_sku")
        with sel_col2:
            states_available = sorted(df_all[df_all["sku_name"] == selected_sku]["state"].unique())
            selected_state = st.selectbox("Select Region/State", states_available, key="scenario_state")
            
        df_ts = df_all[(df_all["sku_name"] == selected_sku) & (df_all["state"] == selected_state)].sort_values(by="observation_date")
        
        if df_ts.empty:
            st.warning("No time-series pricing observations available for this combination.")
        else:
            st.write("---")
            st.markdown("#### 🎚️ Scenario Sliders Adjustments")
            sl_col1, sl_col2 = st.columns(2)
            with sl_col1:
                fuel_shock = st.slider("Fuel Price / Freight Shock (%)", min_value=-20.0, max_value=50.0, value=0.0, step=5.0)
                monsoon_fail = st.slider("Monsoon Failure Index", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            with sl_col2:
                import_duty_adj = st.slider("Import Duty Rate Adjustment (%)", min_value=-100.0, max_value=100.0, value=0.0, step=10.0)
                subsidy_level = st.slider("Targeted Consumer Subsidy (INR/kg)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
                
            latest_row = df_ts.iloc[[-1]].copy()
            if df_clusters is not None:
                latest_row = pd.merge(latest_row, df_clusters[["sku_name", "cluster_id"]], on="sku_name", how="left")
            if "cluster_id" not in latest_row.columns:
                latest_row["cluster_id"] = -1
                
            latest_row["macro_pca_3"] += float(fuel_shock / 100.0) * 0.5
            latest_row["supply_shock_zscore"] -= float(monsoon_fail) * 1.5
            latest_row["supply_shock_zscore"] -= float(import_duty_adj / 100.0) * 0.8
            
            subsidy_qtl = float(subsidy_level) * 100.0
            latest_row["modal_price_per_quintal"] -= subsidy_qtl
            for lag in ["price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d"]:
                latest_row[lag] -= subsidy_qtl
                
            feature_cols = [
                "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
                "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
                "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
                "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
            ]
            cat_cols = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]
            
            X_scenario = latest_row[feature_cols].copy()
                
            if "conformal" in models:
                for col in cat_cols:
                    X_scenario[col] = X_scenario[col].fillna("Missing").astype(str)
                num_cols = [f for f in feature_cols if f not in cat_cols]
                for col in num_cols:
                    X_scenario[col] = X_scenario[col].astype(float)
                y_pred, y_pis = models["conformal"].predict_interval(X_scenario)
                p10 = float(y_pis[0, 0, 0])
                p50 = float(y_pred[0])
                p90 = float(y_pis[0, 1, 0])
            else:
                for col in cat_cols:
                    X_scenario[col] = X_scenario[col].fillna("Missing").astype(str).astype("category")
                p10 = float(models["p10"].predict(X_scenario)[0])
                p50 = float(models["p50"].predict(X_scenario)[0])
                p90 = float(models["p90"].predict(X_scenario)[0])
                
            hist_prices = df_ts["modal_price_per_quintal"].tolist()
            med_f, p10_f, p90_f, raw_samples = forecast_price_trajectories(selected_sku, hist_prices, prediction_length=28)
            
            med_f = med_f - subsidy_qtl
            p10_f = p10_f - subsidy_qtl
            p90_f = p90_f - subsidy_qtl
            raw_samples = raw_samples - subsidy_qtl
            
            macro_shift = float(fuel_shock / 100.0) * 150.0 + float(monsoon_fail) * 250.0 + float(import_duty_adj / 100.0) * 80.0
            med_f = np.maximum(med_f + macro_shift, 100.0)
            p10_f = np.maximum(p10_f + macro_shift, 100.0)
            p90_f = np.maximum(p90_f + macro_shift, 100.0)
            raw_samples = np.maximum(raw_samples + macro_shift, 100.0)
            
            breach_risk = compute_projected_breach_risk(raw_samples, p90)
            
            st.write("---")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Simulated Ceiling Price", f"₹{p90:.2f}")
            with m_col2:
                st.metric("Simulated Midpoint Price", f"₹{p50:.2f}")
            with m_col3:
                if breach_risk > 50.0:
                    st.metric("Projected Breach Risk (Week 4)", f"{breach_risk:.1f}%", delta="CRITICAL RISK ALERT", delta_color="inverse")
                else:
                    st.metric("Projected Breach Risk (Week 4)", f"{breach_risk:.1f}%", delta="COMPLIANT")
                    
            st.markdown("#### 📈 Price Scenario & 4-Week Risk Forecast Path")
            hist_dates = df_ts["observation_date"].tolist()
            forecast_dates = pd.date_range(start=hist_dates[-1] + pd.Timedelta(days=1), periods=28, freq="D").tolist()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_dates, y=hist_prices, mode="lines+markers", name="Historical Price", line=dict(color="#38bdf8", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[hist_dates[-1]] + forecast_dates, y=[p90] * 29, mode="lines", name="Scenario Ceiling (p90)", line=dict(color="#f87171", width=1.5, dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=[hist_dates[-1]] + forecast_dates, y=[p10] * 29, mode="lines", name="Scenario Floor (p10)", line=dict(color="#34d399", width=1.5, dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=med_f, mode="lines", name="Forecasted Price (Median)", line=dict(color="#818cf8", width=2.5)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=p90_f, mode="lines", line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=p10_f, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(129, 140, 248, 0.2)", name="Forecast Confidence Band (80%)"
            ))
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#334155", title="Date"), yaxis=dict(gridcolor="#334155", title="Price (INR/Qtl)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=40, b=0), height=500
            )
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: BATCH RISK UPLOADER ---
with tab_upload:
    st.markdown("### 📤 Batch Compliance Auditor & Transaction Uploader")
    st.write("Upload a CSV file containing transaction price observations to evaluate alignment against fair price bounds.")
    
    # 1. Template Layout
    sample_df = pd.DataFrame([
        {"sku_name": "Tomato", "state": "Uttar Pradesh", "district": "Varanasi", "market_mandi": "Varanasi Mandi", "sku_variety": "Desi", "observed_price": 1450.0},
        {"sku_name": "Tomato", "state": "Uttar Pradesh", "district": "Varanasi", "market_mandi": "Varanasi Mandi", "sku_variety": "Desi", "observed_price": 950.0},
        {"sku_name": "Potato", "state": "Maharashtra", "district": "Pune", "market_mandi": "Pune Mandi", "sku_variety": "Local", "observed_price": 1800.0}
    ])
    st.markdown("**Expected CSV Columns Formatting Structure:**")
    st.dataframe(sample_df, use_container_width=True)
    
    uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"], key="batch_risk_csv")
    
    if uploaded_file is not None:
        st.write("---")
        df_res = pd.DataFrame()
        
        # 2. Process File via API, fallback to Local
        with st.spinner("Analyzing batch pricing compliance..."):
            try:
                res = requests.post(
                    API_RISK_URL, 
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}, 
                    timeout=10
                )
                if res.status_code == 200:
                    df_res = pd.DataFrame(res.json())
                else:
                    st.error(f"FastAPI Backend Error: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.info("FastAPI Backend Server offline. Processing using local conformal model pipeline...")
                try:
                    df_up = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                    df_res = predict_local_batch(df_up)
                except Exception as ex:
                    st.error(f"Error parsing CSV: {str(ex)}")
                    
        # 3. Present Results
        if not df_res.empty:
            # Metrics cards
            high_count = len(df_res[df_res["risk_rating"] == "HIGH RISK"])
            med_count = len(df_res[df_res["risk_rating"] == "MEDIUM RISK"])
            low_count = len(df_res[df_res["risk_rating"] == "LOW RISK"])
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Total Transactions", str(len(df_res)))
            with mc2:
                st.metric("High Risk Alerts", str(high_count), delta=f"{high_count} breaches", delta_color="inverse")
            with mc3:
                st.metric("Medium Risk Alerts", str(med_count))
            with mc4:
                st.metric("Low Risk Tiers", str(low_count))
                
            # Columns Layout for graph & table
            g_col1, g_col2 = st.columns([1, 2])
            
            with g_col1:
                st.markdown("#### Risk Distribution")
                # Plotly Donut Chart
                labels = ["HIGH RISK", "MEDIUM RISK", "LOW RISK"]
                values = [high_count, med_count, low_count]
                colors = ["#ef4444", "#f59e0b", "#10b981"]
                
                # Filter out zero entries
                active_labels = [l for l, v in zip(labels, values) if v > 0]
                active_values = [v for v in values if v > 0]
                active_colors = [c for c, v in zip(colors, values) if v > 0]
                
                if active_values:
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=active_labels,
                        values=active_values,
                        hole=.4,
                        marker=dict(colors=active_colors)
                    )])
                    fig_donut.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=280,
                        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
                else:
                    st.write("No active risk distributions.")
                    
            with g_col2:
                st.markdown("#### Audited Transactions Audit Table")
                def highlight_risk(row):
                    r = row["risk_rating"]
                    if r == "HIGH RISK":
                        return ["background-color: #7f1d1d; color: #fecaca; font-weight: bold"] * len(row)
                    elif r == "MEDIUM RISK":
                        return ["background-color: #78350f; color: #fef3c7"] * len(row)
                    else:
                        return ["background-color: #064e3b; color: #d1fae5"] * len(row)
                        
                st.dataframe(
                    df_res.style.apply(highlight_risk, axis=1),
                    use_container_width=True,
                    height=300
                )
                
            st.write("---")
            # Export report
            csv_report = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Audited Risk Report CSV",
                data=csv_report,
                file_name="casper_audited_pricing_report.csv",
                mime="text/csv"
            )
