import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional
import structlog

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger if hasattr(structlog, "stdlib") else structlog.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

DEFAULT_DB_PATH = "data/audit_log.db"

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the SQLite auditing database and creates the audit_logs table.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                timestamp TEXT,
                sku_id TEXT,
                region TEXT,
                model_version TEXT,
                feature_snapshot_hash TEXT,
                observed_price REAL,
                computed_band TEXT,
                anomaly_type TEXT,
                llm_verdict_json TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()

def log_audit_event(
    sku_id: str,
    region: str,
    model_version: str,
    feature_snapshot_hash: str,
    observed_price: float,
    computed_band: Dict[str, float],
    anomaly_type: str,
    llm_verdict_json: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH
) -> None:
    """
    Persists structured compliance event records to SQLite auditing database,
    and outputs a structured JSON log message via structlog.
    """
    # 1. Initialize DB if not present
    init_db(db_path)
    
    timestamp = datetime.utcnow().isoformat()
    computed_band_str = json.dumps(computed_band)
    llm_verdict_str = json.dumps(llm_verdict_json)
    
    # 2. Persist to SQLite
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (
                timestamp, sku_id, region, model_version, feature_snapshot_hash,
                observed_price, computed_band, anomaly_type, llm_verdict_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, sku_id, region, model_version, feature_snapshot_hash,
            observed_price, computed_band_str, anomaly_type, llm_verdict_str
        ))
        conn.commit()
    finally:
        conn.close()
        
    # 3. Log to stdout in structured JSON via structlog
    logger.info(
        "compliance_audit_event",
        timestamp=timestamp,
        sku_id=sku_id,
        region=region,
        model_version=model_version,
        feature_snapshot_hash=feature_snapshot_hash,
        observed_price=observed_price,
        computed_band=computed_band,
        anomaly_type=anomaly_type,
        llm_verdict_json=llm_verdict_json
    )

def main():
    # Simple CLI verification test
    init_db()
    
    mock_band = {"p10": 2800.0, "p50": 3000.0, "p90": 3300.0}
    mock_verdict = {
        "severity_rating": "HIGH",
        "probable_cause": "Freight spike",
        "recommended_action": "Audit ledger"
    }
    
    log_audit_event(
        sku_id="Onion",
        region="Maharashtra",
        model_version="mapie_conformal_v1.0",
        feature_snapshot_hash="sha256_ab89c897f2",
        observed_price=3500.0,
        computed_band=mock_band,
        anomaly_type="PRICE_GOUGING_ALERT",
        llm_verdict_json=mock_verdict
    )

if __name__ == "__main__":
    main()
