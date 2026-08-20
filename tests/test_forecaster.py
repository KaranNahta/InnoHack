import os
import pytest
import numpy as np

from src.models.chronos_forecaster import (
    forecast_price_trajectories,
    compute_projected_breach_risk,
    statistical_forecast_fallback
)

def test_statistical_forecast_fallback():
    # Verify statistical fallback behaves correctly
    hist = [2000.0, 2100.0, 2200.0, 2300.0, 2400.0]
    med_f, p10_f, p90_f, raw_samples = statistical_forecast_fallback(hist, prediction_length=28, num_samples=15)
    
    assert isinstance(med_f, np.ndarray)
    assert len(med_f) == 28
    assert len(p10_f) == 28
    assert len(p90_f) == 28
    assert raw_samples.shape == (15, 28)
    
    # Assert values are positive
    assert np.all(med_f > 0.0)
    assert np.all(p10_f <= p90_f)

def test_forecast_price_trajectories():
    # Verify that the forecaster returns forecast arrays of length 28
    hist = [1200.0, 1250.0, 1220.0, 1260.0, 1290.0, 1310.0, 1300.0]
    med_f, p10_f, p90_f, raw_samples = forecast_price_trajectories("Potato", hist, prediction_length=28)
    
    assert isinstance(med_f, np.ndarray)
    assert len(med_f) == 28
    assert len(p10_f) == 28
    assert len(p90_f) == 28
    assert raw_samples.ndim == 2
    assert raw_samples.shape[1] == 28
    assert np.all(p10_f <= p90_f)

def test_compute_projected_breach_risk():
    # Verify projected breach risk calculations
    # Let's mock a raw samples matrix of shape (10, 28)
    # 6 paths end above ceiling (3500.0), 4 paths end below
    raw_samples = np.zeros((10, 28))
    # Fill last column index 27
    raw_samples[:6, -1] = 3800.0
    raw_samples[6:, -1] = 3200.0
    
    risk = compute_projected_breach_risk(raw_samples, ceiling_price=3500.0)
    assert isinstance(risk, float)
    assert risk == 60.0 # 6 out of 10 = 60.0%
    
    # All below
    risk_zero = compute_projected_breach_risk(raw_samples, ceiling_price=4000.0)
    assert risk_zero == 0.0
    
    # All above
    risk_all = compute_projected_breach_risk(raw_samples, ceiling_price=3000.0)
    assert risk_all == 100.0
