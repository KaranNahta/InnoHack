"""
CASPER-Gov: Stacking Regressor Ensemble Pipeline
================================================
Combines LightGBM, GradientBoosting / XGBoost, and Random Forest base estimators
with a Ridge regression meta-learner to produce robust calibrated price estimates.
"""

from __future__ import annotations

import logging
import sys
from typing import List

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ensemble_stacking")

FEATURE_COLS = [
    "price_lag_7d", "price_lag_14d", "price_lag_30d", "price_lag_90d",
    "volatility_7d", "volatility_30d", "seasonal_index", "supply_shock_zscore", "is_harvest_season",
    "macro_pca_1", "macro_pca_2", "macro_pca_3", "macro_pca_4", "macro_pca_5",
    "sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"
]
CAT_COLS = ["sku_name", "state", "district", "market_mandi", "sku_variety", "cluster_id"]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]


def get_stacking_pipeline(random_state: int = 42) -> Pipeline:
    """
    Constructs a scikit-learn Pipeline with ColumnTransformer preprocessor
    and StackingRegressor meta-learner.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ],
        remainder="drop",
    )

    base_estimators = [
        ("rf", RandomForestRegressor(n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(n_estimators=60, learning_rate=0.08, max_depth=4, random_state=random_state)),
        ("lgb", lgb.LGBMRegressor(n_estimators=70, learning_rate=0.06, num_leaves=20, random_state=random_state, verbose=-1)),
    ]

    meta_learner = Ridge(alpha=1.0)

    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=3,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("stacking", stacking),
        ]
    )

    return pipeline
