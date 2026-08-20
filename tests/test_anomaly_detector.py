"""
Unit tests for src/models/anomaly_detector.py
==============================================
Covers:
  - AnomalyAlert dataclass construction and serialisation.
  - AnomalyDetector._prepare_features feature shapes and content.
  - AnomalyDetector.fit / is_fitted lifecycle.
  - AnomalyDetector.detect_price_gouging: columns present, severity in [0,1],
    and correct detection of an extreme price spike.
  - AnomalyDetector.detect_cartel_spikes: no false positives on stable data,
    correct detection of a synchronised multi-mandi spike, severity bounds,
    and min_vendors threshold enforcement.
  - AnomalyDetector.save / .load round-trip via tmp_path.
  - detect_anomalies() pipeline: file outputs exist, IsolationForest persisted,
    JSON report contains both PRICE_GOUGING_ALERT and CARTEL_BEHAVIOR_FLAG types.
"""

from __future__ import annotations

import json
import os
from typing import List

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.models.anomaly_detector import (
    AnomalyAlert,
    AnomalyDetector,
    detect_anomalies,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_stable_df(
    n_days: int = 20,
    n_mandis: int = 4,
    base_price: float = 3000.0,
    sku: str = "Rice",
    state: str = "Punjab",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Build a synthetic DataFrame with stable prices (small Gaussian noise).
    All mandis follow the same price distribution with no intentional spikes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-08-01", periods=n_days, freq="D")
    mandis = [f"Mandi_{chr(65 + i)}" for i in range(n_mandis)]  # Mandi_A … Mandi_D

    records = []
    for dt in dates:
        for mandi in mandis:
            records.append(
                {
                    "observation_date": dt,
                    "sku_name": sku,
                    "state": state,
                    "market_mandi": mandi,
                    "modal_price_per_quintal": base_price + rng.normal(0, 5),
                    "arrival_quantity_tonnes": 100.0 + rng.normal(0, 2),
                }
            )

    return pd.DataFrame(records).sort_values(
        ["sku_name", "market_mandi", "observation_date"]
    )


def _make_gouging_df(
    n_days: int = 40,
    sku: str = "Wheat",
    state: str = "Haryana",
    spike_mandi: str = "Mandi_A",
    spike_day_idx: int = 35,
    spike_price: float = 9000.0,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Dataset with one mandi that has an extreme price spike on a specific day.
    All other observations are in the normal range (~3000 ± 5).
    """
    df = _make_stable_df(n_days=n_days, n_mandis=4, base_price=3000.0, sku=sku, state=state, seed=seed)
    df = df.reset_index(drop=True)

    spike_date = pd.date_range("2026-08-01", periods=n_days, freq="D")[spike_day_idx]
    mask = (df["market_mandi"] == spike_mandi) & (df["observation_date"] == spike_date)
    df.loc[mask, "modal_price_per_quintal"] = spike_price
    return df


def _make_cartel_df(
    n_days: int = 20,
    sku: str = "Onion",
    state: str = "Maharashtra",
    cartel_mandis: List[str] = None,
    cartel_day_idx: int = 15,
    cartel_price: float = 6000.0,
    seed: int = 2,
) -> pd.DataFrame:
    """
    Dataset where a subset of mandis spike together on one day.
    Remaining mandis and other days stay at normal prices.
    """
    if cartel_mandis is None:
        cartel_mandis = ["Mandi_B", "Mandi_C", "Mandi_D"]

    df = _make_stable_df(n_days=n_days, n_mandis=4, base_price=3000.0, sku=sku, state=state, seed=seed)
    df = df.reset_index(drop=True)

    spike_date = pd.date_range("2026-08-01", periods=n_days, freq="D")[cartel_day_idx]
    mask = df["market_mandi"].isin(cartel_mandis) & (df["observation_date"] == spike_date)
    df.loc[mask, "modal_price_per_quintal"] = cartel_price
    return df


# ---------------------------------------------------------------------------
# AnomalyAlert tests
# ---------------------------------------------------------------------------

class TestAnomalyAlert:
    def test_construction_and_fields(self):
        alert = AnomalyAlert(
            observation_id="obs_001",
            sku_name="Rice",
            region="Punjab",
            vendor_or_mandi="Mandi_A",
            observed_price=4500.0,
            anomaly_type="PRICE_GOUGING",
            severity_score=0.87,
            details={"z_score": 3.2},
        )
        assert alert.observation_id == "obs_001"
        assert alert.anomaly_type == "PRICE_GOUGING"
        assert 0.0 <= alert.severity_score <= 1.0

    def test_to_dict_round_trip(self):
        alert = AnomalyAlert(
            observation_id="obs_002",
            sku_name="Wheat",
            region="Haryana",
            vendor_or_mandi="Mandi_B",
            observed_price=3200.0,
            anomaly_type="CARTEL_SPIKE",
            severity_score=0.55,
            details={"synchronized_vendors_count": 3},
        )
        d = alert.to_dict()
        assert isinstance(d, dict)
        assert d["sku_name"] == "Wheat"
        assert d["details"]["synchronized_vendors_count"] == 3

    def test_json_serialisable(self):
        alert = AnomalyAlert(
            observation_id="obs_003",
            sku_name="Tomato",
            region="Gujarat",
            vendor_or_mandi="Mandi_C",
            observed_price=2100.0,
            anomaly_type="ARTIFICIAL_SCARCITY",
            severity_score=0.42,
            details={},
        )
        # Should not raise
        json_str = json.dumps(alert.to_dict())
        assert "Tomato" in json_str


# ---------------------------------------------------------------------------
# AnomalyDetector._prepare_features tests
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_shape_matches_input(self):
        df = _make_stable_df(n_days=10, n_mandis=3)
        detector = AnomalyDetector()
        features = detector._prepare_features(df)
        assert features.shape == (len(df), 3)

    def test_column_names(self):
        df = _make_stable_df(n_days=10, n_mandis=2)
        detector = AnomalyDetector()
        features = detector._prepare_features(df)
        assert set(features.columns) == {"pct_change_1d", "dev_from_regional_median", "arrival_quantity"}

    def test_no_nan_values(self):
        df = _make_stable_df(n_days=15, n_mandis=3)
        # Introduce some NaN arrivals – should be imputed
        df.loc[df.index[:5], "arrival_quantity_tonnes"] = np.nan
        detector = AnomalyDetector()
        features = detector._prepare_features(df)
        assert features.isna().sum().sum() == 0

    def test_first_pct_change_per_group_is_zero(self):
        """First observation per (sku, mandi) must have pct_change_1d == 0 (fillna)."""
        df = _make_stable_df(n_days=10, n_mandis=2)
        detector = AnomalyDetector()
        features = detector._prepare_features(df)
        # Verify at least some zeros from the first observation in each group
        assert (features["pct_change_1d"] == 0).any()


# ---------------------------------------------------------------------------
# AnomalyDetector.fit tests
# ---------------------------------------------------------------------------

class TestFit:
    def test_is_fitted_after_fit(self):
        df = _make_stable_df()
        detector = AnomalyDetector()
        assert not detector.is_fitted
        detector.fit(df)
        assert detector.is_fitted

    def test_fit_returns_self(self):
        df = _make_stable_df()
        detector = AnomalyDetector()
        result = detector.fit(df)
        assert result is detector

    def test_iso_forest_is_sklearn_instance(self):
        df = _make_stable_df()
        detector = AnomalyDetector()
        detector.fit(df)
        assert isinstance(detector.iso_forest, IsolationForest)


# ---------------------------------------------------------------------------
# AnomalyDetector.detect_price_gouging tests
# ---------------------------------------------------------------------------

class TestDetectPriceGouging:
    def test_output_columns_present(self):
        df = _make_stable_df()
        detector = AnomalyDetector()
        result = detector.detect_price_gouging(df)
        for col in ("anomaly_raw_score", "is_gouging", "gouging_severity"):
            assert col in result.columns, f"Missing column: {col}"

    def test_severity_in_unit_interval(self):
        df = _make_stable_df()
        detector = AnomalyDetector()
        result = detector.detect_price_gouging(df)
        assert (result["gouging_severity"] >= 0.0).all()
        assert (result["gouging_severity"] <= 1.0).all()

    def test_input_df_not_mutated(self):
        df = _make_stable_df()
        original_cols = list(df.columns)
        detector = AnomalyDetector()
        detector.detect_price_gouging(df)
        assert list(df.columns) == original_cols

    def test_auto_fit_when_not_fitted(self):
        """detect_price_gouging should auto-fit and set is_fitted = True."""
        df = _make_stable_df()
        detector = AnomalyDetector()
        assert not detector.is_fitted
        detector.detect_price_gouging(df)
        assert detector.is_fitted

    def test_extreme_spike_flagged(self):
        """
        An 80%+ price spike in one mandi (while others are stable) must be
        identified as price gouging by the Isolation Forest.
        """
        df = _make_gouging_df(n_days=40, spike_price=9000.0, spike_day_idx=35)
        detector = AnomalyDetector(contamination=0.05)
        result = detector.detect_price_gouging(df)

        spike_date = pd.date_range("2026-08-01", periods=40, freq="D")[35]
        spike_row = result[
            (result["market_mandi"] == "Mandi_A")
            & (result["observation_date"] == spike_date)
        ]
        assert len(spike_row) == 1
        assert bool(spike_row["is_gouging"].iloc[0]), (
            "Extreme price spike was not flagged as gouging by IsolationForest."
        )
        assert spike_row["gouging_severity"].iloc[0] > 0.5

    def test_stable_data_low_false_positive_rate(self):
        """On purely stable data the fraction of flagged observations must
        stay close to the contamination parameter."""
        df = _make_stable_df(n_days=50, n_mandis=4)
        detector = AnomalyDetector(contamination=0.05)
        result = detector.detect_price_gouging(df)
        fpr = result["is_gouging"].mean()
        # Allow a 2x margin around the contamination level
        assert fpr <= 0.10, f"False-positive rate {fpr:.2%} too high on stable data."

    def test_gouging_severity_highest_at_spike(self):
        """The extreme spike row must have the highest severity score."""
        df = _make_gouging_df(n_days=40, spike_price=9000.0, spike_day_idx=35)
        detector = AnomalyDetector(contamination=0.05)
        result = detector.detect_price_gouging(df)

        spike_date = pd.date_range("2026-08-01", periods=40, freq="D")[35]
        spike_severity = result.loc[
            (result["market_mandi"] == "Mandi_A")
            & (result["observation_date"] == spike_date),
            "gouging_severity",
        ].iloc[0]
        max_severity = result["gouging_severity"].max()
        assert spike_severity == pytest.approx(max_severity, abs=0.05), (
            "Spike observation does not have the maximum severity score."
        )


# ---------------------------------------------------------------------------
# AnomalyDetector.detect_cartel_spikes tests
# ---------------------------------------------------------------------------

class TestDetectCartelSpikes:
    def test_no_alerts_on_stable_data(self):
        """Purely stable data (tiny noise) must not generate any cartel alerts."""
        df = _make_stable_df(n_days=20, n_mandis=4)
        detector = AnomalyDetector()
        alerts = detector.detect_cartel_spikes(df)
        assert len(alerts) == 0, (
            f"Expected no cartel alerts on stable data; got {len(alerts)}."
        )

    def test_cartel_spike_detected(self):
        """Three mandis spiking together must trigger at least one CARTEL_SPIKE alert."""
        df = _make_cartel_df(
            n_days=20,
            cartel_mandis=["Mandi_B", "Mandi_C", "Mandi_D"],
            cartel_day_idx=15,
            cartel_price=6000.0,
        )
        detector = AnomalyDetector()
        alerts = detector.detect_cartel_spikes(df, std_threshold=2.5, min_vendors=3)
        assert len(alerts) > 0, "Expected at least one cartel-spike alert."

    def test_alert_type_is_cartel_spike(self):
        df = _make_cartel_df()
        detector = AnomalyDetector()
        alerts = detector.detect_cartel_spikes(df, std_threshold=2.5, min_vendors=3)
        for alert in alerts:
            assert alert.anomaly_type == "CARTEL_SPIKE"

    def test_alert_severity_in_unit_interval(self):
        df = _make_cartel_df()
        detector = AnomalyDetector()
        alerts = detector.detect_cartel_spikes(df)
        for alert in alerts:
            assert 0.0 <= alert.severity_score <= 1.0, (
                f"severity_score {alert.severity_score} out of [0, 1]."
            )

    def test_alert_details_populated(self):
        df = _make_cartel_df()
        detector = AnomalyDetector()
        alerts = detector.detect_cartel_spikes(df, min_vendors=3)
        for alert in alerts:
            assert "synchronized_vendors_count" in alert.details
            assert alert.details["synchronized_vendors_count"] >= 3
            assert "vendor_list" in alert.details
            assert "avg_z_score" in alert.details

    def test_min_vendors_threshold_enforced(self):
        """
        With min_vendors=4, a 3-mandi spike must NOT trigger any alert.
        """
        df = _make_cartel_df(
            cartel_mandis=["Mandi_B", "Mandi_C", "Mandi_D"],  # exactly 3
            cartel_day_idx=15,
            cartel_price=6000.0,
        )
        detector = AnomalyDetector()
        alerts = detector.detect_cartel_spikes(df, min_vendors=4)
        assert len(alerts) == 0, (
            "A 3-mandi spike should not qualify when min_vendors=4."
        )

    def test_correct_mandis_in_alert(self):
        """The vendor_list in alert details must contain the cartel mandis."""
        cartel_mandis = ["Mandi_B", "Mandi_C", "Mandi_D"]
        df = _make_cartel_df(cartel_mandis=cartel_mandis, cartel_day_idx=15, cartel_price=6000.0)
        detector = AnomalyDetector()
        alerts = detector.detect_cartel_spikes(df, min_vendors=3)
        assert len(alerts) > 0
        # Collect all vendors across all alerts
        all_vendors: set = set()
        for alert in alerts:
            all_vendors.update(alert.details["vendor_list"])
        for mandi in cartel_mandis:
            assert mandi in all_vendors, f"{mandi} not found in any alert vendor_list."

    def test_returns_list_of_anomaly_alerts(self):
        df = _make_cartel_df()
        detector = AnomalyDetector()
        alerts = detector.detect_cartel_spikes(df)
        assert isinstance(alerts, list)
        for alert in alerts:
            assert isinstance(alert, AnomalyAlert)

    def test_input_df_not_mutated(self):
        df = _make_cartel_df()
        original_len = len(df)
        original_cols = list(df.columns)
        detector = AnomalyDetector()
        detector.detect_cartel_spikes(df)
        assert len(df) == original_len
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# AnomalyDetector persistence (save / load) tests
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        df = _make_stable_df()
        detector = AnomalyDetector()
        detector.fit(df)
        path = str(tmp_path / "detector.joblib")
        detector.save(path)
        assert os.path.exists(path)

    def test_load_returns_anomaly_detector(self, tmp_path):
        df = _make_stable_df()
        detector = AnomalyDetector()
        detector.fit(df)
        path = str(tmp_path / "detector.joblib")
        detector.save(path)

        loaded = AnomalyDetector.load(path)
        assert isinstance(loaded, AnomalyDetector)

    def test_loaded_detector_is_fitted(self, tmp_path):
        df = _make_stable_df()
        detector = AnomalyDetector()
        detector.fit(df)
        path = str(tmp_path / "detector.joblib")
        detector.save(path)

        loaded = AnomalyDetector.load(path)
        assert loaded.is_fitted

    def test_loaded_detector_produces_same_scores(self, tmp_path):
        """Scores from a loaded detector must match the original's scores."""
        df = _make_stable_df()
        detector = AnomalyDetector()
        result_original = detector.detect_price_gouging(df)

        path = str(tmp_path / "detector.joblib")
        detector.save(path)
        loaded = AnomalyDetector.load(path)
        result_loaded = loaded.detect_price_gouging(df)

        np.testing.assert_array_almost_equal(
            result_original["anomaly_raw_score"].values,
            result_loaded["anomaly_raw_score"].values,
            decimal=6,
        )

    def test_column_config_preserved_after_load(self, tmp_path):
        df = _make_stable_df()
        detector = AnomalyDetector(
            price_col="modal_price_per_quintal",
            region_col="state",
        )
        detector.fit(df)
        path = str(tmp_path / "detector.joblib")
        detector.save(path)
        loaded = AnomalyDetector.load(path)
        assert loaded.price_col == "modal_price_per_quintal"
        assert loaded.region_col == "state"


# ---------------------------------------------------------------------------
# detect_anomalies() pipeline tests
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_data(tmp_path):
    """
    Produce a Parquet features file and a vendor registry Parquet file
    containing injected gouging (Mandi_A, day 10) and cartel spikes
    (Mandi_B/C/D, day 14) – mirrors the structure of test_anomalies.py.
    """
    dates = pd.date_range("2026-08-01", periods=15, freq="D")
    mandis = ["Mandi A", "Mandi B", "Mandi C", "Mandi D"]
    records = []
    for dt in dates:
        for mandi in mandis:
            price = 3000.0
            # Extreme individual spike → PRICE_GOUGING_ALERT
            if dt.strftime("%Y-%m-%d") == "2026-08-10" and mandi == "Mandi A":
                price = 5400.0
            # Synchronised cartel spike → CARTEL_BEHAVIOR_FLAG
            if dt.strftime("%Y-%m-%d") == "2026-08-14" and mandi in [
                "Mandi B", "Mandi C", "Mandi D"
            ]:
                price = 3900.0
            records.append(
                {
                    "observation_date": dt,
                    "sku_name": "Rice",
                    "state": "Punjab",
                    "market_mandi": mandi,
                    "modal_price_per_quintal": price,
                    "arrival_quantity_tonnes": 100.0,
                }
            )
    features_path = str(tmp_path / "features.parquet")
    pd.DataFrame(records).to_parquet(features_path, index=False)

    vendor_records = [
        {"vendor_id": f"VEND_{m}", "region": "Punjab", "sku": "Rice"}
        for m in ["A", "B", "C", "D"]
    ]
    vendor_path = str(tmp_path / "vendors.parquet")
    pd.DataFrame(vendor_records).to_parquet(vendor_path, index=False)

    return features_path, vendor_path


class TestDetectAnomaliesPipeline:
    def test_model_file_created(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        model_path = str(tmp_path / "iso.joblib")
        report_path = str(tmp_path / "report.json")
        detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=model_path,
            report_save_path=report_path,
        )
        assert os.path.exists(model_path), "IsolationForest joblib file not created."

    def test_report_file_created(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        report_path = str(tmp_path / "report.json")
        detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=str(tmp_path / "iso.joblib"),
            report_save_path=report_path,
        )
        assert os.path.exists(report_path), "Anomaly JSON report not created."

    def test_persisted_model_is_isolation_forest(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        model_path = str(tmp_path / "iso.joblib")
        detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=model_path,
            report_save_path=str(tmp_path / "report.json"),
        )
        model = joblib.load(model_path)
        assert isinstance(model, IsolationForest)

    def test_report_is_valid_json_list(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        report_path = str(tmp_path / "report.json")
        detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=str(tmp_path / "iso.joblib"),
            report_save_path=report_path,
        )
        with open(report_path) as fh:
            data = json.load(fh)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_return_value_matches_report(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        report_path = str(tmp_path / "report.json")
        returned = detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=str(tmp_path / "iso.joblib"),
            report_save_path=report_path,
        )
        with open(report_path) as fh:
            persisted = json.load(fh)
        assert len(returned) == len(persisted)

    def test_price_gouging_alert_present(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        alerts = detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=str(tmp_path / "iso.joblib"),
            report_save_path=str(tmp_path / "report.json"),
        )
        types = [a["anomaly_type"] for a in alerts]
        assert "PRICE_GOUGING_ALERT" in types, (
            "PRICE_GOUGING_ALERT not present in pipeline output."
        )

    def test_cartel_behavior_flag_present(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        alerts = detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=str(tmp_path / "iso.joblib"),
            report_save_path=str(tmp_path / "report.json"),
        )
        types = [a["anomaly_type"] for a in alerts]
        assert "CARTEL_BEHAVIOR_FLAG" in types, (
            "CARTEL_BEHAVIOR_FLAG not present in pipeline output."
        )

    def test_gouging_date_2026_08_10(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        alerts = detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=str(tmp_path / "iso.joblib"),
            report_save_path=str(tmp_path / "report.json"),
        )
        gouging_dates = [
            a["date"]
            for a in alerts
            if a["anomaly_type"] == "PRICE_GOUGING_ALERT"
        ]
        assert "2026-08-10" in gouging_dates, (
            "Expected gouging alert on 2026-08-10; got: " + str(gouging_dates)
        )

    def test_cartel_date_2026_08_14(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        alerts = detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=str(tmp_path / "iso.joblib"),
            report_save_path=str(tmp_path / "report.json"),
        )
        cartel_dates = [
            a["date"]
            for a in alerts
            if a["anomaly_type"] == "CARTEL_BEHAVIOR_FLAG"
        ]
        assert "2026-08-14" in cartel_dates, (
            "Expected cartel alert on 2026-08-14; got: " + str(cartel_dates)
        )

    def test_all_records_have_required_keys(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        alerts = detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=str(tmp_path / "iso.joblib"),
            report_save_path=str(tmp_path / "report.json"),
        )
        required_keys = {
            "observation_id", "date", "sku_name", "state",
            "market_mandi", "observed_price", "anomaly_type",
            "severity_score", "details",
        }
        for alert in alerts:
            missing = required_keys - set(alert.keys())
            assert not missing, f"Alert record missing keys: {missing}"

    def test_severity_scores_in_unit_interval(self, pipeline_data, tmp_path):
        features_path, vendor_path = pipeline_data
        alerts = detect_anomalies(
            features_path=features_path,
            vendor_path=vendor_path,
            model_save_path=str(tmp_path / "iso.joblib"),
            report_save_path=str(tmp_path / "report.json"),
        )
        for alert in alerts:
            s = alert["severity_score"]
            assert 0.0 <= s <= 1.0, f"severity_score {s} out of [0, 1]."
