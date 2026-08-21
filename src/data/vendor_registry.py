"""
CASPER-Gov: Wholesaler & Retailer Vendor Registry Engine
========================================================
Generates synthetic and production vendor registry metadata for compliance tracking,
license verification, and regional mandate attribution.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vendor_registry")

REGIONS = [
    "Uttar Pradesh", "Punjab", "Haryana", "Maharashtra", "Gujarat",
    "Karnataka", "Madhya Pradesh", "Rajasthan", "Tamil Nadu", "Andhra Pradesh",
    "Bihar", "West Bengal", "Kerala", "Telangana", "Odisha"
]
RETAILER_TYPES = ["wholesaler", "retail_chain", "independent"]
COMMODITIES = [
    "Rice", "Wheat", "Potato", "Onion", "Tomato", "Gram Dal",
    "Mustard Oil", "Sugar", "Turmeric", "Cotton", "Maize",
    "Soyabean", "Groundnut", "Moong Dal", "Urad Dal", "Apple"
]



def generate_vendor_registry(n_vendors: int = 50, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic vendor registry with vendor_id, licensed_status,
    retailer_type, region, and registered_skus.
    """
    rng = np.random.RandomState(random_seed)
    records = []

    for i in range(1, n_vendors + 1):
        vendor_id = f"VEND_{i:04d}"
        licensed = bool(rng.rand() > 0.10) # 90% licensed
        r_type = str(rng.choice(RETAILER_TYPES))
        region = str(rng.choice(REGIONS))
        
        # Pick 1 to 3 registered SKUs
        n_skus = rng.randint(1, 4)
        chosen_skus = list(rng.choice(COMMODITIES, size=n_skus, replace=False))
        registered_skus = ", ".join(chosen_skus)

        records.append({
            "vendor_id": vendor_id,
            "licensed_status": licensed,
            "retailer_type": r_type,
            "region": region,
            "registered_skus": registered_skus,
        })

    return pd.DataFrame(records)


def save_vendor_registry(df: pd.DataFrame, output_path: str = "data/raw/vendor_registry.parquet") -> None:
    """
    Saves vendor registry to Parquet.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d vendor records to %s", len(df), output_path)
