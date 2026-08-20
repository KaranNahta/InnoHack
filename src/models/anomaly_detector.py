"""
CASPER-Gov: Cartel & Anomaly Detection Engine
=============================================
Implements two complementary detection strategies:
  1. Point-level price gouging via Isolation Forest (individual mandi outliers).
  2. Multi-mandi synchronized cartel spike detection via rolling z-score analysis.

Exports
-------
AnomalyAlert       : Dataclass representing a single detected anomaly event.
AnomalyDetector    : Stateful detector class (fit / detect / persist).
detect_anomalies   : High-level pipeline entry-point (used by test_anomalies.py).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("anomaly_detector")

# ---------------------------------------------------------------------------
# Schema constants – single source of truth for column names used across the
# project (see build_features.py, test_anomalies.py, etc.)
# ---------------------------------------------------------------------------
_DEFAULT_PRICE_COL = "modal_price_per_quintal"
_DEFAULT_ARRIVAL_COL = "arrival_quantity_tonnes"
_DEFAULT_DATE_COL = "observation_date"
_DEFAULT_MANDI_COL = "market_mandi"
_DEFAULT_SKU_COL = "sku_name"
_DEFAULT_REGION_COL = "state"  # project uses 'state' as the regional grouper


# ---------------------------------------------------------------------------
# AnomalyAlert dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnomalyAlert:
    """A single detected anomaly event emitted by the AnomalyDetector."""

    observation_id: str
    sku_name: str
    region: str
    vendor_or_mandi: str
    observed_price: float
    anomaly_type: str   # 'PRICE_GOUGING', 'CARTEL_SPIKE', 'ARTIFICIAL_SCARCITY'
    severity_score: float   # normalised 0.0-1.0
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Cartel & Anomaly Detection Engine for CASPER-Gov.

    Parameters
    ----------
    contamination : float
        Expected proportion of outliers in the training set (passed to
        IsolationForest). Defaults to 0.05.
    price_col : str
        Name of the price column in incoming DataFrames.
    arrival_col : str
        Name of the arrival-volume column.
    date_col : str
        Name of the observation date column.
    mandi_col : str
        Name of the market / mandi identifier column.
    sku_col : str
        Name of the SKU / commodity identifier column.
    region_col : str
        Name of the regional grouper column (e.g. 'state', 'region').
    """

    def __init__(
        self,
        contamination: float = 0.05,
        price_col: str = _DEFAULT_PRICE_COL,
        arrival_col: str = _DEFAULT_ARRIVAL_COL,
        date_col: str = _DEFAULT_DATE_COL,
        mandi_col: str = _DEFAULT_MANDI_COL,
        sku_col: str = _DEFAULT_SKU_COL,
        region_col: str = _DEFAULT_REGION_COL,
    ) -> None:
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
        )
        self.is_fitted: bool = False

        # Column name configuration
        self.price_col = price_col
        self.arrival_col = arrival_col
        self.date_col = date_col
        self.mandi_col = mandi_col
        self.sku_col = sku_col
        self.region_col = region_col

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract a 3-column statistical feature matrix for the Isolation Forest.

        Features
        --------
        pct_change_1d           : Day-over-day price percentage change per
                                  (sku, mandi) time series.
        dev_from_regional_median: Absolute deviation of each observation's price
                                  from the daily regional median price of that SKU.
        arrival_quantity        : Arrival volume (tonnes), median-imputed.
        """
        price = df[self.price_col]

        pct_change = (
            df.groupby([self.sku_col, self.mandi_col])[self.price_col]
            .pct_change()
            .fillna(0)
        )

        regional_median = df.groupby(
            [self.sku_col, self.region_col, self.date_col]
        )[self.price_col].transform("median")

        arrival = df[self.arrival_col].fillna(df[self.arrival_col].median())

        features = pd.DataFrame(
            {
                "pct_change_1d": pct_change.values,
                "dev_from_regional_median": (price - regional_median).values,
                "arrival_quantity": arrival.values,
            },
            index=df.index,
        )
        return features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit the Isolation Forest on historical baseline data.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataset. Must contain all configured column names.

        Returns
        -------
        self
        """
        logger.info("Fitting Isolation Forest on %d observations ...", len(df))
        X = self._prepare_features(df)
        self.iso_forest.fit(X)
        self.is_fitted = True
        logger.info("Isolation Forest fitted.")
        return self

    # ------------------------------------------------------------------
    # Detection: individual price gouging
    # ------------------------------------------------------------------

    def detect_price_gouging(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag individual vendor / mandi price gouging via Isolation Forest.

        If the detector has not been fitted yet it will be auto-fitted on df
        (convenient for single-dataset workflows; in production always call
        fit on a separate historical baseline first).

        Returns
        -------
        pd.DataFrame
            A copy of df with three extra columns:

            * anomaly_raw_score  - raw IsolationForest decision score
              (lower values = more anomalous).
            * is_gouging         - boolean flag; True when the forest
              labels the point as an outlier (score == -1).
            * gouging_severity   - float in [0, 1]; 1.0 is most severe.
        """
        if not self.is_fitted:
            logger.warning(
                "AnomalyDetector not fitted; auto-fitting on the provided dataset. "
                "For production use, fit on a separate historical baseline."
            )
            self.fit(df)

        X = self._prepare_features(df)
        df = df.copy()

        raw_scores = self.iso_forest.decision_function(X)
        df["anomaly_raw_score"] = raw_scores
        df["is_gouging"] = self.iso_forest.predict(X) == -1

        # Normalise: lower raw score -> higher severity -> closer to 1.0
        score_range = raw_scores.max() - raw_scores.min() + 1e-6
        df["gouging_severity"] = np.clip(
            1.0 - (raw_scores - raw_scores.min()) / score_range, 0.0, 1.0
        )
        return df

    # ------------------------------------------------------------------
    # Detection: multi-mandi cartel synchronisation
    # ------------------------------------------------------------------

    def detect_cartel_spikes(
        self,
        df: pd.DataFrame,
        std_threshold: float = 2.5,
        min_vendors: int = 3,
    ) -> List[AnomalyAlert]:
        """
        Detect multi-mandi synchronised price spikes within the same region.

        Strategy
        --------
        1. For each (sku, mandi) pair compute a 14-day rolling z-score.
        2. Filter observations whose z-score exceeds std_threshold.
        3. Group spikes by (date, region, sku); if >= min_vendors distinct
           mandis spike simultaneously the event is flagged as a cartel spike.

        Parameters
        ----------
        df            : Input DataFrame sorted by date.
        std_threshold : Z-score threshold above which a single observation is
                        considered a spike (default 2.5).
        min_vendors   : Minimum number of distinct mandis that must spike
                        on the same date/region/sku to qualify as a cartel
                        event (default 3).

        Returns
        -------
        List[AnomalyAlert]
            One alert per (mandi, date) row that is part of a qualifying
            cartel event.
        """
        alerts: List[AnomalyAlert] = []
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values([self.sku_col, self.mandi_col, self.date_col])

        # --- Rolling 14-day statistics per (sku, mandi) time-series ----------
        grp = df.groupby([self.sku_col, self.mandi_col])[self.price_col]

        df["price_mean_14d"] = grp.transform(
            lambda x: x.rolling(14, min_periods=3).mean()
        )
        df["price_std_14d"] = grp.transform(
            lambda x: x.rolling(14, min_periods=3).std()
        ).fillna(1.0)
        df["z_score"] = (df[self.price_col] - df["price_mean_14d"]) / (
            df["price_std_14d"] + 1e-6
        )

        # --- Filter to spike observations ------------------------------------
        spikes = df[df["z_score"] >= std_threshold]

        # --- Detect synchronised cartel events -------------------------------
        grouped = spikes.groupby([self.date_col, self.region_col, self.sku_col])

        for (obs_date, region, sku), group in grouped:
            unique_vendors = group[self.mandi_col].nunique()
            if unique_vendors < min_vendors:
                continue

            avg_spike = float(group["z_score"].mean())

            for _, row in group.iterrows():
                alerts.append(
                    AnomalyAlert(
                        observation_id=str(
                            row.get(
                                "id",
                                f"{obs_date}_{sku}_{row[self.mandi_col]}",
                            )
                        ),
                        sku_name=sku,
                        region=region,
                        vendor_or_mandi=row[self.mandi_col],
                        observed_price=float(row[self.price_col]),
                        anomaly_type="CARTEL_SPIKE",
                        severity_score=min(avg_spike / 5.0, 1.0),
                        details={
                            "synchronized_vendors_count": unique_vendors,
                            "vendor_list": group[self.mandi_col].tolist(),
                            "avg_z_score": round(avg_spike, 2),
                            "observation_date": str(obs_date),
                        },
                    )
                )

        logger.info(
            "detect_cartel_spikes: %d cartel-spike alert(s) generated.", len(alerts)
        )
        return alerts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str = "models/isolation_forest.joblib") -> None:
        """Persist the entire AnomalyDetector instance to disk via joblib."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("AnomalyDetector saved to %s", filepath)

    @classmethod
    def load(cls, filepath: str = "models/isolation_forest.joblib") -> "AnomalyDetector":
        """Deserialise an AnomalyDetector from a joblib file."""
        detector: AnomalyDetector = joblib.load(filepath)
        logger.info("AnomalyDetector loaded from %s", filepath)
        return detector


# ---------------------------------------------------------------------------
# Pipeline entry-point  (imported by test_anomalies.py)
# ---------------------------------------------------------------------------

def detect_anomalies(
    features_path: str,
    vendor_path: str,
    model_save_path: str = "models/isolation_forest.joblib",
    report_save_path: str = "reports/price_anomalies.json",
    contamination: float = 0.05,
    cartel_std_threshold: float = 2.5,
    cartel_min_vendors: int = 3,
) -> List[Dict[str, Any]]:
    """
    High-level anomaly-detection pipeline for CASPER-Gov.

    Reads price-feature data and a vendor registry, runs both the Isolation
    Forest price-gouging detector and the rolling z-score cartel detector,
    then persists the fitted model and a JSON anomaly report.

    Parameters
    ----------
    features_path      : Path to a Parquet file with time-series price features.
    vendor_path        : Path to a Parquet file with the vendor registry.
    model_save_path    : Where to write the fitted IsolationForest (joblib).
    report_save_path   : Where to write the anomaly report (JSON list).
    contamination      : IsolationForest contamination parameter.
    cartel_std_threshold : Z-score threshold for a single-mandi spike.
    cartel_min_vendors : Min mandis spiking simultaneously to flag a cartel.

    Returns
    -------
    List[Dict[str, Any]]
        Combined list of anomaly records (both PRICE_GOUGING_ALERT and
        CARTEL_BEHAVIOR_FLAG entries), also written to report_save_path.
    """
    logger.info("Loading features from %s ...", features_path)
    df = pd.read_parquet(features_path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values(["sku_name", "market_mandi", "observation_date"])

    # ------------------------------------------------------------------
    # 1.  Fit detector & score every observation for price gouging
    # ------------------------------------------------------------------
    detector = AnomalyDetector(contamination=contamination)
    df_scored = detector.detect_price_gouging(df)

    # ------------------------------------------------------------------
    # 2.  Persist the underlying IsolationForest (bare sklearn model)
    #     so that test_anomalies.py can assert isinstance(model, IsolationForest)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    joblib.dump(detector.iso_forest, model_save_path)
    logger.info("IsolationForest saved to %s", model_save_path)

    # ------------------------------------------------------------------
    # 3.  Build PRICE_GOUGING_ALERT records from individual outliers
    # ------------------------------------------------------------------
    anomaly_records: List[Dict[str, Any]] = []

    gouging_rows = df_scored[df_scored["is_gouging"]]
    for _, row in gouging_rows.iterrows():
        anomaly_records.append(
            {
                "observation_id": str(
                    row.get(
                        "id",
                        f"{row['observation_date']}_{row['sku_name']}_{row['market_mandi']}",
                    )
                ),
                "date": pd.Timestamp(row["observation_date"]).strftime("%Y-%m-%d"),
                "sku_name": row["sku_name"],
                "state": row.get("state", ""),
                "market_mandi": row["market_mandi"],
                "observed_price": float(row["modal_price_per_quintal"]),
                "anomaly_type": "PRICE_GOUGING_ALERT",
                "severity_score": round(float(row["gouging_severity"]), 4),
                "details": {
                    "anomaly_raw_score": round(float(row["anomaly_raw_score"]), 6),
                },
            }
        )

    logger.info("Price-gouging alerts: %d", len(anomaly_records))

    # ------------------------------------------------------------------
    # 4.  Detect cartel events via rolling z-score
    # ------------------------------------------------------------------
    cartel_alerts = detector.detect_cartel_spikes(
        df,
        std_threshold=cartel_std_threshold,
        min_vendors=cartel_min_vendors,
    )

    for alert in cartel_alerts:
        anomaly_records.append(
            {
                "observation_id": alert.observation_id,
                "date": str(alert.details.get("observation_date", ""))[:10],
                "sku_name": alert.sku_name,
                "state": alert.region,
                "market_mandi": alert.vendor_or_mandi,
                "observed_price": alert.observed_price,
                "anomaly_type": "CARTEL_BEHAVIOR_FLAG",
                "severity_score": round(alert.severity_score, 4),
                "details": alert.details,
            }
        )

    logger.info("Cartel-behavior flags: %d", len(cartel_alerts))

    # ------------------------------------------------------------------
    # 5.  Write JSON report
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(report_save_path) or ".", exist_ok=True)
    with open(report_save_path, "w", encoding="utf-8") as fh:
        json.dump(anomaly_records, fh, indent=2, default=str)

    logger.info(
        "Anomaly report written to %s  (%d total alerts).",
        report_save_path,
        len(anomaly_records),
    )
    return anomaly_records
