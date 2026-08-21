import os
import sys

# Ensure workspace root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import io
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    go = None
    HAS_PLOTLY = False

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


# ─────────────────────────────────────────────────────────────
# AUTO-BOOTSTRAP: Generate datasets if parquet files are missing
# (e.g. first deploy on Streamlit Cloud where data/ is gitignored)
# ─────────────────────────────────────────────────────────────
def bootstrap_data_if_needed():
    """
    Auto-generates all required parquet data if any are missing.
    Runs on Streamlit Cloud where data/ is not committed to git.
    """
    import os
    from datetime import datetime as dt

    needs_bootstrap = not all(os.path.exists(p) for p in [
        TEST_FEAT_PATH, VENDORS_PATH, CLUSTERS_PATH
    ])

    if not needs_bootstrap:
        return

    with st.spinner("⏳ First-time setup: generating Agmarknet data pipeline (takes ~30 sec)..."):
        try:
            import sys
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

            from src.data.agmarknet_ingest import (
                fetch_daily_mandi_prices,
                normalize_agmarknet_schema,
                validate_agmarknet_schema,
                save_raw_agmarknet_data,
            )
            from src.data.vendor_registry import generate_vendor_registry, save_vendor_registry
            from src.features.build_features import transform_features
            from src.models.goods_clustering import run_goods_clustering

            # 1. Ingest 60 days from Agmarknet
            end = dt.now().strftime("%Y-%m-%d")
            start = (dt.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
            df_raw = fetch_daily_mandi_prices(start_date=start, end_date=end)
            df_norm = normalize_agmarknet_schema(df_raw)
            df_clean = validate_agmarknet_schema(df_norm)
            os.makedirs("data/raw/agmarknet", exist_ok=True)
            save_raw_agmarknet_data(df_clean, output_dir="data/raw/agmarknet")
            df_clean.to_parquet("data/raw/agmarknet_combined.parquet", index=False)

            # 2. Vendor registry
            df_v = generate_vendor_registry(n_vendors=100)
            save_vendor_registry(df_v, output_path=VENDORS_PATH)

            # 3. Feature engineering
            df_feat = transform_features(df_clean)
            os.makedirs("data/features", exist_ok=True)
            df_feat.to_parquet(TEST_FEAT_PATH, index=False)
            df_feat.to_parquet("data/features/train_features.parquet", index=False)

            # 4. Clustering
            run_goods_clustering(input_features_path="data/features/train_features.parquet")

            st.success("✅ Data pipeline bootstrapped successfully!")
        except Exception as ex:
            st.warning(f"⚠️ Auto-bootstrap encountered an issue: {ex}. Using demo fallback data.")
            _generate_demo_fallback_data()


def _generate_demo_fallback_data():
    """Generates minimal in-memory demo data if the full pipeline fails."""
    import os
    import numpy as np

    rng = np.random.RandomState(42)
    skus = ["Rice", "Wheat", "Potato", "Onion", "Tomato", "Gram Dal",
            "Mustard Oil", "Sugar", "Maize", "Moong Dal", "Urad Dal",
            "Turmeric", "Cotton", "Groundnut", "Soyabean", "Apple"]
    states = ["Uttar Pradesh", "Punjab", "Maharashtra", "Gujarat", "Karnataka",
              "Madhya Pradesh", "Rajasthan", "Tamil Nadu", "Andhra Pradesh",
              "Bihar", "West Bengal", "Kerala", "Telangana", "Haryana", "Odisha"]
    mandis = {s: [f"{s[:4].strip()} Mandi A", f"{s[:4].strip()} Mandi B"] for s in states}

    records = []
    today = pd.Timestamp.today().normalize()
    for sku in skus:
        base = rng.uniform(800, 4000)
        for state in states:
            for mandi in mandis[state]:
                obs = float(base * rng.uniform(0.9, 1.3))
                p90 = float(base * 1.1)
                records.append({
                    "observation_date": today.strftime("%Y-%m-%d"),
                    "sku_name": sku, "state": state, "market_mandi": mandi,
                    "vendor_id": f"VEND_{rng.randint(1, 100):04d}",
                    "modal_price_per_quintal": obs, "observed_price": obs,
                    "p10_floor": float(base * 0.85), "p50_mid": float(base),
                    "p90_ceiling": p90,
                    "compliance_status": "CEILING_BREACHED" if obs > p90 else "WITHIN_BAND",
                })

    df = pd.DataFrame(records)
    os.makedirs("data/features", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    df.to_parquet(TEST_FEAT_PATH, index=False)
    df[["vendor_id", "state", "sku_name"]].rename(columns={"state": "region", "sku_name": "registered_skus"}).to_parquet(VENDORS_PATH, index=False)
    df[["sku_name"]].drop_duplicates().assign(cluster_id=0).to_parquet(CLUSTERS_PATH, index=False)


# Run bootstrap on every cold start
bootstrap_data_if_needed()

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

# Global Sidebar Filter Configuration
st.sidebar.markdown("### 🎛️ Regulatory Control Filters")

all_skus = sorted(df_monitor["sku_name"].unique().tolist()) if not df_monitor.empty else []
all_states = sorted(df_monitor["state"].unique().tolist()) if not df_monitor.empty else []
all_statuses = sorted(df_monitor["compliance_status"].unique().tolist()) if not df_monitor.empty else []

selected_skus = st.sidebar.multiselect(
    "Commodity SKUs",
    all_skus,
    default=all_skus,
    help="Filter commodities across all surveillance tabs"
)

selected_states = st.sidebar.multiselect(
    "Regions / States",
    all_states,
    default=all_states,
    help="Filter geographical administrative regions"
)

selected_status = st.sidebar.multiselect(
    "Compliance Status",
    all_statuses,
    default=all_statuses,
    help="Filter by CEILING_BREACHED, ELEVATED_PRICE, or WITHIN_BAND"
)

# Apply global filters
df_filtered = df_monitor[
    (df_monitor["sku_name"].isin(selected_skus if selected_skus else all_skus)) &
    (df_monitor["state"].isin(selected_states if selected_states else all_states)) &
    (df_monitor["compliance_status"].isin(selected_status if selected_status else all_statuses))
]

st.sidebar.markdown("---")
st.sidebar.caption("⚡ **CASPER-Gov v1.0** · Real-Time Market Surveillance Engine")

# Tabs
tab_mandi, tab_audit, tab_blockchain, tab_legal_ai, tab_scenario, tab_upload = st.tabs([
    "🏪 Mandi+ Live Cards",
    "📋 Live Audit Monitor", 
    "🛡️ Cryptographic Audit Ledger",
    "🤖 AI Statutory Legal Counsel",
    "⚡ Scenario Planning & Risk Forecasting",
    "📤 Batch Risk Uploader"
])


# ─────────────────────────────────────────────────────────────
# TAB 0: MANDI+ LIVE COMMODITY CARDS (Slides 6–9)
# ─────────────────────────────────────────────────────────────
with tab_mandi:
    st.markdown("### 🏪 Mandi+ — Live Commodity Intelligence Cards")
    st.caption("Real-time price band status, compliance badges, and hoarding risk signals per commodity")

    if df_filtered.empty:
        st.warning("No records match the current sidebar filter selection.")
    else:
        df_cards = df_filtered.copy()


        # Aggregate: one row per SKU (latest date, averaged across mandis)
        price_col = "observed_price" if "observed_price" in df_cards.columns else "modal_price_per_quintal"
        p50_col = "p50_mid" if "p50_mid" in df_cards.columns else "p50_midpoint"
        p10_col = "p10_floor"
        p90_col = "p90_ceiling"

        df_agg = (
            df_cards.groupby("sku_name").agg(
                observed_price=(price_col, "mean"),
                p10_floor=(p10_col, "mean"),
                p50_mid=(p50_col, "mean"),
                p90_ceiling=(p90_col, "mean"),
                num_mandis=("market_mandi", "nunique"),
                num_breaches=("compliance_status", lambda x: (x == "CEILING_BREACHED").sum()),
                state=("state", "first"),
            ).reset_index()
        )

        # Summary metrics strip
        mc1, mc2, mc3, mc4 = st.columns(4)
        total_skus = len(df_agg)
        breach_skus = len(df_agg[df_agg["num_breaches"] > 0])
        total_mandis_card = int(df_agg["num_mandis"].sum())
        avg_compliance = round((1 - breach_skus / max(total_skus, 1)) * 100, 1)
        with mc1:
            st.metric("Commodities Monitored", total_skus)
        with mc2:
            st.metric("Commodities with Breaches", breach_skus, delta=f"{breach_skus} alerts", delta_color="inverse")
        with mc3:
            st.metric("Total Mandis", total_mandis_card)
        with mc4:
            st.metric("Overall Compliance", f"{avg_compliance}%")

        st.write("---")

        # Commodity cards — 3 per row
        CARDS_PER_ROW = 3
        skus = df_agg["sku_name"].tolist()
        for row_start in range(0, len(skus), CARDS_PER_ROW):
            row_skus = skus[row_start: row_start + CARDS_PER_ROW]
            cols = st.columns(CARDS_PER_ROW)
            for col_idx, sku in enumerate(row_skus):
                row = df_agg[df_agg["sku_name"] == sku].iloc[0]
                observed = float(row["observed_price"])
                p10 = float(row["p10_floor"])
                p50 = float(row["p50_mid"])
                p90 = float(row["p90_ceiling"])
                n_mandis = int(row["num_mandis"])
                n_breaches = int(row["num_breaches"])

                # Status classification
                if observed > p90:
                    status_color = "#ef4444"
                    status_label = "🔴 CEILING BREACHED"
                    badge_bg = "#7f1d1d"
                    badge_fg = "#fecaca"
                elif observed > p50:
                    status_color = "#f59e0b"
                    status_label = "🟡 ELEVATED PRICE"
                    badge_bg = "#78350f"
                    badge_fg = "#fef3c7"
                else:
                    status_color = "#10b981"
                    status_label = "🟢 COMPLIANT"
                    badge_bg = "#064e3b"
                    badge_fg = "#d1fae5"

                # Hoarding risk: proxied from supply shock (if available) or breach ratio
                hoarding_pct = min(100, int(n_breaches / max(n_mandis, 1) * 100))
                if hoarding_pct >= 50:
                    hoard_chip = "🚨 High Hoarding Risk"
                    hoard_color = "#ef4444"
                elif hoarding_pct >= 20:
                    hoard_chip = "⚠️ Moderate Risk"
                    hoard_color = "#f59e0b"
                else:
                    hoard_chip = "✅ Low Risk"
                    hoard_color = "#10b981"

                # Gauge fill percent (0–100%) relative to band
                band_range = max(p90 - p10, 1.0)
                gauge_pct = min(100, max(0, int((observed - p10) / band_range * 100)))

                with cols[col_idx]:
                    st.markdown(f"""
<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid {status_color}44;
            border-radius: 16px; padding: 20px; margin-bottom: 12px;
            box-shadow: 0 4px 24px {status_color}22;">

  <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
    <div>
      <h3 style="margin:0; font-size:1.2rem; color:#f1f5f9; font-family:Inter,sans-serif;">
        🌾 {sku}
      </h3>
      <p style="margin:2px 0 0; font-size:0.75rem; color:#94a3b8;">{row['state']} · {n_mandis} Mandis</p>
    </div>
    <span style="background:{badge_bg}; color:{badge_fg}; font-size:0.65rem;
                 font-weight:700; padding:4px 10px; border-radius:999px; white-space:nowrap;">
      {status_label}
    </span>
  </div>

  <!-- Observed price -->
  <div style="text-align:center; margin-bottom:14px;">
    <div style="font-size:2rem; font-weight:800; color:{status_color}; font-family:Inter,sans-serif;">
      ₹{observed:,.0f}
    </div>
    <div style="font-size:0.72rem; color:#64748b;">Observed Avg Price (INR/Qtl)</div>
  </div>

  <!-- Price band gauge bar -->
  <div style="margin-bottom:14px;">
    <div style="display:flex; justify-content:space-between; font-size:0.68rem; color:#94a3b8; margin-bottom:4px;">
      <span>Floor ₹{p10:,.0f}</span><span>Fair ₹{p50:,.0f}</span><span>Ceiling ₹{p90:,.0f}</span>
    </div>
    <div style="background:#334155; border-radius:999px; height:8px; position:relative; overflow:hidden;">
      <div style="background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444);
                  width:100%; height:100%; position:absolute; opacity:0.3; border-radius:999px;"></div>
      <div style="background:{status_color}; height:100%; width:6px; position:absolute;
                  left:{gauge_pct}%; transform:translateX(-50%); border-radius:999px;
                  box-shadow: 0 0 6px {status_color};"></div>
    </div>
  </div>

  <!-- Hoarding risk chip -->
  <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem;">
    <span style="color:{hoard_color}; font-weight:600;">{hoard_chip}</span>
    <span style="color:#64748b;">{n_breaches}/{n_mandis} mandis breached</span>
  </div>

</div>
""", unsafe_allow_html=True)

                    # 7-stage price estimate button
                    if st.button(f"🔍 Deep Estimate", key=f"btn_estimate_{sku}_{col_idx}_{row_start}"):
                        with st.spinner(f"Running 7-stage pipeline for {sku}..."):
                            try:
                                res = requests.post(
                                    f"http://{API_HOST}:8000/api/v1/price-estimate",
                                    json={"sku_name": sku, "state": row["state"]},
                                    timeout=15,
                                )
                                if res.status_code == 200:
                                    data = res.json()
                                    stage7 = data["stages"]["stage_7_final"]
                                    critic = data["stages"]["stage_6_critic"]
                                    shap_d = data["stages"]["stage_4_shap_drivers"]
                                    st.success(f"**Critic Decision:** {critic['decision']}  |  **Risk:** {stage7['risk_level']}")
                                    st.json({
                                        "final_p10": stage7["p10_floor"],
                                        "final_p50": stage7["p50_midpoint"],
                                        "final_p90": stage7["p90_ceiling"],
                                        "compliance": stage7["compliance_status"],
                                        "top_driver": shap_d[0]["feature"] if shap_d else "N/A",
                                    })

                                    # Provide instant PDF generation for non-compliant anomalies
                                    if stage7["compliance_status"] != "WITHIN_BAND":
                                        from src.utils.pdf_exporter import build_enforcement_pdf
                                        notice_payload = {
                                            "notice_id": f"ENF-{sku[:3].upper()}-{datetime.utcnow().strftime('%Y%m%d%H%M')}",
                                            "severity_rating": stage7["risk_level"],
                                            "sku_name": sku,
                                            "target_entity": f"{row['state']} Wholesale Mandis",
                                            "region": row["state"],
                                            "observed_price": stage7["observed_price"],
                                            "fair_price_ceiling": stage7["p90_ceiling"],
                                            "price_deviation_pct": stage7["breach_percentage"],
                                            "top_cost_drivers": shap_d,
                                            "legal_citations": data["stages"]["stage_5_legal_precedents"],
                                            "recommended_action": "Issue Statutory Price Show-Cause Notice under Section 3 ECA 1955",
                                            "draft_notice_text": f"OFFICIAL STATUTORY NOTICE: Surveillance indicates an abnormal price deviation of +{stage7['breach_percentage']:.1f}% for {sku} in {row['state']}. Supply registers and cost audit are demanded within 48 hours."
                                        }
                                        pdf_bytes = build_enforcement_pdf(notice_payload)
                                        st.download_button(
                                            label="📄 Download Official Court-Ready Notice (PDF)",
                                            data=pdf_bytes,
                                            file_name=f"statutory_enforcement_notice_{sku}_{datetime.utcnow().strftime('%Y%m%d')}.pdf",
                                            mime="application/pdf",
                                            key=f"dl_pdf_{sku}_{col_idx}_{row_start}"
                                        )
                                else:
                                    st.error(f"API Error {res.status_code}")
                            except Exception as ex:
                                st.warning(f"API offline — direct model fallback: {str(ex)[:80]}")

        # -------------------------------------------------------------
        # Inter-Mandi Cartel Collusion Network Visualizer (Slide 14)
        # -------------------------------------------------------------
        st.write("---")
        st.markdown("### 🕸️ Inter-Mandi Price Collusion & Cartel Network Analysis")
        st.markdown("Visualizes synchronized vendor/mandi pricing spikes. Connected red clusters identify syndicates engaging in non-competitive pricing collusion exceeding competition thresholds ($r > 0.75$).")

        from src.dashboard.components.cartel_graph import build_cartel_network_figure
        c_col1, c_col2 = st.columns([3, 2])
        with c_col1:
            cartel_sku = st.selectbox("Select Commodity for Cartel Topology Analysis", all_skus, key="cartel_sku_select")
            cartel_fig, cartel_cliques = build_cartel_network_figure(df_monitor, selected_sku=cartel_sku)
            if cartel_fig is not None:
                st.plotly_chart(cartel_fig, use_container_width=True)
            else:
                st.info("📊 Graph visualization requires `plotly` (`pip install plotly`).")
        with c_col2:
            st.markdown(f"#### 🚨 Detected Collusion Syndicates ({len(cartel_cliques)} Links)")
            if cartel_cliques:
                st.dataframe(pd.DataFrame(cartel_cliques), use_container_width=True)
                st.warning(f"⚠️ **Antitrust Alert:** {len(cartel_cliques)} mandi pairs exhibit statistically anomalous price synchronization under Section 3(3)(a) Competition Act 2002.")
            else:
                st.success("✅ **Competitive Pricing:** No abnormal inter-mandi price synchronization detected.")

# --- TAB 1: AUDIT MONITOR ---
with tab_audit:
    st.markdown("### 📋 Active Market Pricing Audit Table & Metrics")


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

    # -------------------------------------------------------------
    # Cryptographic Audit Chain Integrity Verification (Slide 15)
    # -------------------------------------------------------------
    st.write("---")
    st.markdown("### 🔐 Cryptographic Audit Chain & Tamper-Evident Ledger (Slide 15)")
    st.markdown("Every regulatory price estimation, anomaly alert, and enforcement action is cryptographically sealed using SHA-256 block-chaining (`hash_n = SHA256(hash_{n-1} + timestamp + payload)`).")

    from src.audit.logger import verify_audit_trail, get_audit_logs
    audit_res = verify_audit_trail()
    recent_logs = get_audit_logs(limit=10)

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        if audit_res.get("chain_valid"):
            st.metric("Blockchain Verification", "VALID & INTACT", delta="Zero Tampering")
        else:
            st.metric("Blockchain Verification", "TAMPER DETECTED", delta="Integrity Compromised", delta_color="inverse")
    with ac2:
        st.metric("Total Cryptographic Blocks", str(audit_res.get("total_records", 0)))
    with ac3:
        latest_root = str(audit_res.get("latest_root_hash", "0000..."))[:16] + "..."
        st.metric("Latest Merkle Root Hash", latest_root)

    with st.expander("📜 View Cryptographic Audit Log Entries (Latest 10 Blocks)", expanded=False):
        if recent_logs:
            df_logs = pd.DataFrame(recent_logs)[["id", "timestamp", "sku_id", "region", "anomaly_type", "observed_price", "prev_hash", "entry_hash"]]
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("No audit entries recorded yet. Generate an enforcement notice or run predictions to append blocks.")

# ─────────────────────────────────────────────────────────────
# TAB 2: CRYPTOGRAPHIC AUDIT BLOCKCHAIN & TAMPER SIMULATOR
# ─────────────────────────────────────────────────────────────
with tab_blockchain:
    st.markdown("### 🛡️ SHA-256 Cryptographic Audit Blockchain & Tamper-Evident Ledger")
    st.caption("Mathematical proof of non-repudiation for statutory tribunals and high courts under Indian Evidence Act §65B.")

    from src.audit.logger import (
        verify_audit_trail, 
        get_audit_logs, 
        simulate_audit_tampering, 
        repair_tampered_audit_trail
    )

    verdict = verify_audit_trail()
    is_valid = verdict.get("chain_valid", False)
    total_blocks = verdict.get("total_records", 0)

    # Top Status Indicators
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        if is_valid:
            st.success(f"🔒 **Status: CHAIN INTACT**\n\nAll {total_blocks} blocks mathematically verified from genesis.")
        else:
            tampered_id = verdict.get("tampered_record_id", "?")
            st.error(f"🚨 **STATUS: TAMPER DETECTED**\n\nBlock #{tampered_id} hash signature mismatch!")
    with b_col2:
        st.metric("Chain Length", f"{total_blocks} Blocks")
    with b_col3:
        st.metric("Hash Algorithm", "SHA-256 (Chained)")
    with b_col4:
        st.metric("Legal Admissibility", "IEA §65B Compliant", delta="Court Ready")

    st.write("---")

    # Interactive Judge Tamper-Proof Demonstration Box
    st.markdown("#### 🧪 Interactive Judge Proof: Live Anti-Tampering Demonstration")
    st.markdown(
        "Demonstrates how the system mathematically catches unauthorized manual alterations in the underlying database. "
        "Clicking **'Simulate Database Tampering'** maliciously modifies an observed price in SQLite without regenerating the hash. "
        "The cryptographic verifier immediately detects the broken signature."
    )

    t_col1, t_col2, t_col3 = st.columns([2, 2, 2])
    with t_col1:
        if st.button("🚨 Simulate Database Tampering", key="btn_tamper_sim", use_container_width=True):
            res = simulate_audit_tampering()
            if res.get("success"):
                st.warning(f"⚠️ {res.get('message')}")
                st.rerun()
            else:
                st.info(res.get("message"))

    with t_col2:
        if st.button("🔍 Cryptographically Verify Chain", key="btn_verify_sim", use_container_width=True):
            res = verify_audit_trail()
            if res.get("chain_valid"):
                st.success(f"✅ Mathematical Proof Verified: {res.get('message', 'Chain valid')}")
            else:
                st.error(f"❌ TAMPER DETECTED! {res.get('message')}")

    with t_col3:
        if st.button("🛡️ Restore & Re-Seal Chain", key="btn_repair_sim", use_container_width=True):
            res = repair_tampered_audit_trail()
            st.success(f"✅ {res.get('message')}")
            st.rerun()

    st.write("---")
    st.markdown("#### 📜 Full Cryptographic Block Ledger")
    all_logs = get_audit_logs(limit=100)
    if all_logs:
        df_all_logs = pd.DataFrame(all_logs)
        display_cols = ["id", "timestamp", "sku_id", "region", "anomaly_type", "observed_price", "prev_hash", "entry_hash"]
        available_cols = [c for c in display_cols if c in df_all_logs.columns]
        st.dataframe(df_all_logs[available_cols], use_container_width=True, height=350)
    else:
        st.info("Ledger initialized. Run a price estimate or scenario to create genesis transactions.")

# ─────────────────────────────────────────────────────────────
# TAB 3: AI STATUTORY LEGAL COUNSEL (RAG Q&A)
# ─────────────────────────────────────────────────────────────
with tab_legal_ai:
    st.markdown("### 🤖 AI Statutory Legal Counsel & Regulatory Precedents Assistant")
    st.caption("Grounded retrieval-augmented legal advisor for Mandi Officers & Enforcement Tribunals (ECA 1955, Competition Act 2002, Legal Metrology).")

    from src.rag.vector_store import RegulatoryVectorStore, STATUTORY_PRECEDENTS

    # Initialise chat history in session state
    if "legal_chat_history" not in st.session_state:
        st.session_state.legal_chat_history = [
            {
                "role": "assistant",
                "content": (
                    "**Welcome to the CASPER-Gov Statutory Legal Intelligence Counsel.**\n\n"
                    "I can answer queries regarding penal directives, price gouging thresholds, cartel collusion laws, and show-cause notice wording. "
                    "How can I assist your enforcement tribunal today?"
                ),
                "sources": []
            }
        ]

    # Pre-canned Quick Questions for Judges
    st.markdown("**⚡ Quick Precedents & Legal Prompts:**")
    qc1, qc2, qc3 = st.columns(3)
    prompt_to_submit = None
    with qc1:
        if st.button("⚖️ What are penal provisions under ECA 1955 for price gouging?", use_container_width=True):
            prompt_to_submit = "What are penal provisions and search/seizure powers under Section 3 and Section 7 of ECA 1955 for price gouging?"
    with qc2:
        if st.button("🧅 How does Competition Act 2002 §3(3)(a) apply to Mandi cartels?", use_container_width=True):
            prompt_to_submit = "How does Section 3(3)(a) of Competition Act 2002 apply to inter-mandi price fixing and vendor collusion?"
    with qc3:
        if st.button("📦 What are mandatory packaging rules under Legal Metrology 2011?", use_container_width=True):
            prompt_to_submit = "What are the price declaration rules and penalties under Legal Metrology Packaged Commodities Rules 2011?"

    # Display existing chat messages
    for msg in st.session_state.legal_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Cited Statutory Precedents & Corpus Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src.get('statute', 'Statute')} ({src.get('section', '')})**: {src.get('title', '')}")
                        st.caption(f"_{src.get('text', '')[:200]}..._")

    # Handle user query
    user_input = st.chat_input("Ask a legal or statutory enforcement question...")
    query = prompt_to_submit or user_input

    if query:
        st.session_state.legal_chat_history.append({"role": "user", "content": query, "sources": []})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving statutory precedents from ChromaDB ONNX store..."):
                try:
                    store = RegulatoryVectorStore()
                    retrieved_docs = store.retrieve_relevant_precedents(query=query, top_k=3)
                except Exception as ex:
                    # Fallback keyword match from static corpus
                    q_lower = query.lower()
                    retrieved_docs = [
                        p for p in STATUTORY_PRECEDENTS 
                        if any(w in p["text"].lower() for w in q_lower.split() if len(w) > 3)
                    ][:3]
                    if not retrieved_docs:
                        retrieved_docs = STATUTORY_PRECEDENTS[:2]

                # Synthesize legal response
                source_bullets = "\n".join([f"- **{d.get('statute', 'Act')} ({d.get('section', '')})**: {d.get('title', '')}" for d in retrieved_docs])
                precedents_context = "\n\n".join([f"**[{d.get('statute', 'Statute')} - {d.get('section', '')}]**\n{d.get('text', '')}" for d in retrieved_docs])
                
                response_text = (
                    f"### ⚖️ Statutory Legal Assessment\n\n"
                    f"Based on the query: **\"{query}\"**, the following statutory provisions and executive directives apply:\n\n"
                    f"{precedents_context}\n\n"
                    f"#### 📌 Enforcement Officer Directives:\n"
                    f"1. **Statutory Authority**: Issue Form-IV Show Cause Notice referencing cited sections.\n"
                    f"2. **Evidence Packaging**: Attach the SHA-256 sealed price distribution certificate under Indian Evidence Act §65B.\n"
                    f"3. **Remedial Timeline**: 72 hours for wholesale vendor response prior to inventory seizure or license suspension."
                )
                
                st.markdown(response_text)
                with st.expander("📚 Cited Statutory Precedents & Corpus Sources", expanded=True):
                    for src in retrieved_docs:
                        st.markdown(f"- **{src.get('statute', 'Statute')} ({src.get('section', '')})**: {src.get('title', '')}")
                        st.caption(f"_{src.get('text', '')[:250]}..._")

                st.session_state.legal_chat_history.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": retrieved_docs
                })

# --- TAB 4: SCENARIO PLANNING ---
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
