"""
CASPER-Gov: Zero-Shot Chronos Forecaster & Trajectory Simulator
==============================================================
Provides price trajectory forecasting and breach risk estimation using
Chronos foundation time-series models with a robust statistical fallback.
"""

from __future__ import annotations

import logging
import sys
from typing import List, Tuple, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("chronos_forecaster")

# Attempt importing chronos / torch lazily or conditionally
_CHRONOS_PIPELINE = None


def get_chronos_pipeline(model_name: str = "amazon/chronos-t5-small"):
    """Lazily load chronos pipeline if dependencies and weights are available."""
    global _CHRONOS_PIPELINE
    if _CHRONOS_PIPELINE is not None:
        return _CHRONOS_PIPELINE
    try:
        import torch
        from chronos import ChronosPipeline
        logger.info("Initializing ChronosPipeline (%s)...", model_name)
        _CHRONOS_PIPELINE = ChronosPipeline.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        return _CHRONOS_PIPELINE
    except Exception as e:
        logger.warning("ChronosPipeline initialization unavailable (%s). Falling back to statistical forecast.", str(e))
        return None


def statistical_forecast_fallback(
    history: List[float],
    prediction_length: int = 28,
    num_samples: int = 20,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Statistical auto-regressive / random walk drift simulation for price paths.
    
    Returns:
      - median_forecast (np.ndarray of shape (prediction_length,))
      - p10_forecast (np.ndarray of shape (prediction_length,))
      - p90_forecast (np.ndarray of shape (prediction_length,))
      - raw_samples (np.ndarray of shape (num_samples, prediction_length))
    """
    rng = np.random.RandomState(random_seed)
    hist_arr = np.array(history, dtype=float)
    if len(hist_arr) == 0:
        hist_arr = np.array([1000.0])
    
    last_price = float(hist_arr[-1])
    
    # Calculate drift and volatility
    if len(hist_arr) > 1:
        returns = np.diff(hist_arr) / np.maximum(hist_arr[:-1], 1e-5)
        mean_drift = float(np.mean(returns))
        volatility = float(np.std(returns))
        if volatility < 1e-4:
            volatility = 0.015
    else:
        mean_drift = 0.0
        volatility = 0.02
        
    # Clip drift to avoid explosive divergence
    mean_drift = np.clip(mean_drift, -0.05, 0.05)
    volatility = np.clip(volatility, 0.005, 0.15)
    
    # Generate Monte Carlo paths
    samples = np.zeros((num_samples, prediction_length), dtype=float)
    for s in range(num_samples):
        price = last_price
        for t in range(prediction_length):
            shock = rng.normal(mean_drift, volatility)
            price = max(price * (1.0 + shock), 10.0)
            samples[s, t] = price
            
    median_f = np.median(samples, axis=0)
    p10_f = np.percentile(samples, 10, axis=0)
    p90_f = np.percentile(samples, 90, axis=0)
    
    return median_f, p10_f, p90_f, samples


def forecast_price_trajectories(
    sku_name: str,
    history: List[float],
    prediction_length: int = 28,
    num_samples: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates forecasted price trajectories for a given commodity price history.
    Uses Chronos pipeline if available, otherwise statistical Monte Carlo simulation.
    """
    pipeline = get_chronos_pipeline()
    
    if pipeline is not None:
        try:
            import torch
            context = torch.tensor(history, dtype=torch.float32)
            forecast = pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
            ) # shape: (1, num_samples, prediction_length)
            samples = forecast[0].numpy()
            median_f = np.median(samples, axis=0)
            p10_f = np.percentile(samples, 10, axis=0)
            p90_f = np.percentile(samples, 90, axis=0)
            return median_f, p10_f, p90_f, samples
        except Exception as e:
            logger.warning("Chronos inference failed: %s. Using statistical fallback.", str(e))
            
    return statistical_forecast_fallback(history, prediction_length, num_samples)


def compute_projected_breach_risk(raw_samples: np.ndarray, ceiling_price: float) -> float:
    """
    Calculates the percentage of trajectory sample paths that breach the ceiling price
    at the end of the forecast horizon (last time step).
    
    Returns:
      - float risk between 0.0 and 100.0 (percentage)
    """
    if raw_samples is None or raw_samples.size == 0:
        return 0.0
    
    # Check breaches at the horizon end (last step)
    final_step_prices = raw_samples[:, -1]
    breaches = np.sum(final_step_prices > ceiling_price)
    total_samples = len(final_step_prices)
    
    if total_samples == 0:
        return 0.0
        
    return float(round((breaches / total_samples) * 100.0, 2))
