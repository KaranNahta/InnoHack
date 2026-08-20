import os
import sys
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging

try:
    import structlog
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
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("audit_logger")

DEFAULT_DB_PATH = "data/audit_log.db"
GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


def compute_entry_hash(
    prev_hash: str,
    timestamp: str,
    sku_id: str,
    region: str,
    model_version: str,
    feature_snapshot_hash: str,
    observed_price: float,
    computed_band_str: str,
    anomaly_type: str,
    llm_verdict_str: str
) -> str:
    """Computes deterministic SHA-256 hash for audit record chaining."""
    payload = f"{prev_hash}|{timestamp}|{sku_id}|{region}|{model_version}|{feature_snapshot_hash}|{observed_price:.2f}|{computed_band_str}|{anomaly_type}|{llm_verdict_str}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the SQLite auditing database with chained cryptographic hash fields.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                sku_id TEXT,
                region TEXT,
                model_version TEXT,
                feature_snapshot_hash TEXT,
                observed_price REAL,
                computed_band TEXT,
                anomaly_type TEXT,
                llm_verdict_json TEXT,
                prev_hash TEXT,
                entry_hash TEXT
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
) -> str:
    """
    Persists structured compliance event records with SHA-256 blockchain-style chaining.
    Returns the entry_hash of the recorded event.
    """
    init_db(db_path)
    
    timestamp = datetime.utcnow().isoformat()
    computed_band_str = json.dumps(computed_band, sort_keys=True)
    llm_verdict_str = json.dumps(llm_verdict_json, sort_keys=True)
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        # Fetch the latest previous hash in the chain
        cursor.execute("SELECT entry_hash FROM audit_logs ORDER BY id DESC LIMIT 1")
        last_row = cursor.fetchone()
        prev_hash = last_row[0] if (last_row and last_row[0]) else GENESIS_HASH

        # Compute current entry hash
        entry_hash = compute_entry_hash(
            prev_hash=prev_hash,
            timestamp=timestamp,
            sku_id=sku_id,
            region=region,
            model_version=model_version,
            feature_snapshot_hash=feature_snapshot_hash,
            observed_price=observed_price,
            computed_band_str=computed_band_str,
            anomaly_type=anomaly_type,
            llm_verdict_str=llm_verdict_str,
        )

        cursor.execute("""
            INSERT INTO audit_logs (
                timestamp, sku_id, region, model_version, feature_snapshot_hash,
                observed_price, computed_band, anomaly_type, llm_verdict_json,
                prev_hash, entry_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, sku_id, region, model_version, feature_snapshot_hash,
            observed_price, computed_band_str, anomaly_type, llm_verdict_str,
            prev_hash, entry_hash
        ))
        conn.commit()
    finally:
        conn.close()
        
    try:
        logger.info(
            "compliance_audit_event",
            timestamp=timestamp,
            sku_id=sku_id,
            region=region,
            model_version=model_version,
            entry_hash=entry_hash,
            prev_hash=prev_hash,
            observed_price=observed_price,
            anomaly_type=anomaly_type,
        )
    except TypeError:
        logger.info(
            "compliance_audit_event: sku=%s region=%s entry_hash=%s prev_hash=%s observed=%.2f type=%s",
            sku_id, region, entry_hash[:12], prev_hash[:12], observed_price, anomaly_type
        )
    return entry_hash


def verify_audit_trail(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Cryptographically verifies the non-tampered integrity of all audit records in the database.
    Recomputes the hash chain from genesis and returns validation verdict.
    """
    if not os.path.exists(db_path):
        return {
            "verified": True,
            "total_records": 0,
            "chain_valid": True,
            "message": "Audit database empty or initialized.",
            "tampered_record_id": None
        }

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, sku_id, region, model_version, feature_snapshot_hash,
                   observed_price, computed_band, anomaly_type, llm_verdict_json,
                   prev_hash, entry_hash
            FROM audit_logs
            ORDER BY id ASC
        """)
        rows = cursor.fetchall()

        if not rows:
            return {
                "verified": True,
                "total_records": 0,
                "chain_valid": True,
                "message": "Zero records in audit log.",
                "tampered_record_id": None
            }

        expected_prev_hash = GENESIS_HASH
        for row in rows:
            (row_id, ts, sku, reg, m_ver, feat_h, obs_p, band_s, anom_t, verdict_s, prev_h, stored_hash) = row
            
            # Check previous hash link
            if prev_h != expected_prev_hash:
                return {
                    "verified": False,
                    "total_records": len(rows),
                    "chain_valid": False,
                    "tampered_record_id": row_id,
                    "message": f"Broken chain link at record ID {row_id}. Expected prev_hash {expected_prev_hash[:8]}..., found {prev_h[:8]}..."
                }

            # Recompute hash
            recomputed = compute_entry_hash(
                prev_hash=prev_h,
                timestamp=ts,
                sku_id=sku,
                region=reg,
                model_version=m_ver,
                feature_snapshot_hash=feat_h,
                observed_price=float(obs_p),
                computed_band_str=band_s,
                anomaly_type=anom_t,
                llm_verdict_str=verdict_s
            )

            if recomputed != stored_hash:
                return {
                    "verified": False,
                    "total_records": len(rows),
                    "chain_valid": False,
                    "tampered_record_id": row_id,
                    "message": f"Cryptographic signature mismatch at record ID {row_id}. Data has been modified!"
                }

            expected_prev_hash = stored_hash

        return {
            "verified": True,
            "total_records": len(rows),
            "chain_valid": True,
            "latest_root_hash": expected_prev_hash,
            "message": f"All {len(rows)} audit records cryptographically verified with unbroken SHA-256 chain."
        }
    finally:
        conn.close()


def get_audit_logs(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent audit log entries formatted as structured dicts."""
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, sku_id, region, model_version, feature_snapshot_hash,
                   observed_price, computed_band, anomaly_type, llm_verdict_json,
                   prev_hash, entry_hash
            FROM audit_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        logs = []
        for r in rows:
            try:
                band_dict = json.loads(r[7]) if r[7] else {}
            except Exception:
                band_dict = {}
            try:
                verdict_dict = json.loads(r[9]) if r[9] else {}
            except Exception:
                verdict_dict = {}

            logs.append({
                "id": r[0],
                "timestamp": r[1],
                "sku_id": r[2],
                "region": r[3],
                "model_version": r[4],
                "feature_snapshot_hash": r[5],
                "observed_price": r[6],
                "computed_band": band_dict,
                "anomaly_type": r[8],
                "llm_verdict": verdict_dict,
                "prev_hash": r[10],
                "entry_hash": r[11],
            })
        return logs
    finally:
        conn.close()


def simulate_audit_tampering(record_id: Optional[int] = None, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Simulates a malicious attack or unapproved modification on an audit record by
    altering observed_price directly in SQLite without updating the cryptographic hash.
    Used for live judge demonstration of tamper-detection.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        if record_id is None:
            cursor.execute("SELECT id, observed_price FROM audit_logs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                return {"success": False, "message": "No audit records found to tamper."}
            target_id, orig_price = row[0], row[1]
        else:
            cursor.execute("SELECT id, observed_price FROM audit_logs WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            if not row:
                return {"success": False, "message": f"Record {record_id} not found."}
            target_id, orig_price = row[0], row[1]

        tampered_price = round(float(orig_price) * 1.85, 2)
        cursor.execute("UPDATE audit_logs SET observed_price = ? WHERE id = ?", (tampered_price, target_id))
        conn.commit()
        logger.warning("Simulated tampering on record %d: price changed from %.2f to %.2f", target_id, orig_price, tampered_price)
        return {
            "success": True,
            "tampered_record_id": target_id,
            "original_price": orig_price,
            "tampered_price": tampered_price,
            "message": f"Record #{target_id} price altered maliciously to ₹{tampered_price}. Hash chain is now compromised."
        }
    finally:
        conn.close()


def repair_tampered_audit_trail(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Recalculates all entry hashes sequentially from genesis to repair a simulated tamper.
    """
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, sku_id, region, model_version, feature_snapshot_hash,
                   observed_price, computed_band, anomaly_type, llm_verdict_json
            FROM audit_logs
            ORDER BY id ASC
        """)
        rows = cursor.fetchall()
        prev_h = GENESIS_HASH
        for r in rows:
            r_id, ts, sku, reg, m_ver, feat_h, obs_p, band_s, anom_t, verdict_s = r
            new_hash = compute_entry_hash(
                prev_hash=prev_h,
                timestamp=ts,
                sku_id=sku,
                region=reg,
                model_version=m_ver,
                feature_snapshot_hash=feat_h,
                observed_price=obs_p,
                computed_band_str=band_s,
                anomaly_type=anom_t,
                llm_verdict_json_str=verdict_s
            )
            cursor.execute("UPDATE audit_logs SET prev_hash = ?, entry_hash = ? WHERE id = ?", (prev_h, new_hash, r_id))
            prev_h = new_hash
        conn.commit()
        return {"success": True, "message": f"Repaired {len(rows)} blocks. Chain integrity restored to valid state."}
    finally:
        conn.close()


def main():
    init_db()
    
    mock_band = {"p10": 2800.0, "p50": 3000.0, "p90": 3300.0}
    mock_verdict = {
        "severity_rating": "HIGH",
        "probable_cause": "Freight spike",
        "recommended_action": "Audit ledger"
    }
    
    h1 = log_audit_event(
        sku_id="Onion",
        region="Maharashtra",
        model_version="mapie_conformal_v1.0",
        feature_snapshot_hash="sha256_ab89c897f2",
        observed_price=3500.0,
        computed_band=mock_band,
        anomaly_type="PRICE_GOUGING_ALERT",
        llm_verdict_json=mock_verdict
    )
    
    h2 = log_audit_event(
        sku_id="Tomato",
        region="Uttar Pradesh",
        model_version="mapie_conformal_v1.0",
        feature_snapshot_hash="sha256_cd45e678f1",
        observed_price=4200.0,
        computed_band=mock_band,
        anomaly_type="ARTIFICIAL_SCARCITY",
        llm_verdict_json=mock_verdict
    )

    verdict = verify_audit_trail()
    print("Verification result:", json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()

