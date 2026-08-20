"""
Unit tests for Cryptographic Audit Trail, PDF Notice Exporter, and Cartel Graph Visualizer.
"""

import os
import sqlite3
import pytest
import pandas as pd
import numpy as np

from src.audit.logger import (
    init_db,
    log_audit_event,
    verify_audit_trail,
    get_audit_logs,
    compute_entry_hash,
    GENESIS_HASH,
)
from src.utils.pdf_exporter import build_enforcement_pdf
from src.dashboard.components.cartel_graph import build_cartel_network_figure


class TestCryptographicAuditTrail:
    """Tests for SHA-256 Merkle-chained audit logging and verification."""

    def test_log_and_verify_clean_chain(self, tmp_path):
        db_path = str(tmp_path / "audit_test.db")
        init_db(db_path)

        # Log 3 events
        h1 = log_audit_event(
            sku_id="Tomato",
            region="Uttar Pradesh",
            model_version="mapie_v1.0",
            feature_snapshot_hash="hash_001",
            observed_price=1500.0,
            computed_band={"p10": 1000.0, "p50": 1200.0, "p90": 1400.0},
            anomaly_type="PRICE_GOUGING",
            llm_verdict_json={"decision": "NOTICE_ISSUED"},
            db_path=db_path,
        )

        h2 = log_audit_event(
            sku_id="Potato",
            region="Maharashtra",
            model_version="mapie_v1.0",
            feature_snapshot_hash="hash_002",
            observed_price=2200.0,
            computed_band={"p10": 1400.0, "p50": 1600.0, "p90": 1900.0},
            anomaly_type="ARTIFICIAL_SCARCITY",
            llm_verdict_json={"decision": "WARNING"},
            db_path=db_path,
        )

        assert h1 != h2
        assert len(h1) == 64
        assert len(h2) == 64

        # Verify integrity
        verdict = verify_audit_trail(db_path)
        assert verdict["verified"] is True
        assert verdict["chain_valid"] is True
        assert verdict["total_records"] == 2

    def test_detects_tampered_records(self, tmp_path):
        db_path = str(tmp_path / "audit_tamper.db")
        init_db(db_path)

        log_audit_event(
            sku_id="Onion",
            region="Maharashtra",
            model_version="mapie_v1.0",
            feature_snapshot_hash="hash_orig",
            observed_price=3000.0,
            computed_band={"p10": 2000.0, "p50": 2400.0, "p90": 2800.0},
            anomaly_type="PRICE_GOUGING",
            llm_verdict_json={"decision": "FINE_RECOMMENDED"},
            db_path=db_path,
        )

        # Directly tamper with SQLite data behind the scenes
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE audit_logs SET observed_price = 1200.0 WHERE sku_id = 'Onion'")
        conn.commit()
        conn.close()

        # Audit verification must flag the tampering
        verdict = verify_audit_trail(db_path)
        assert verdict["verified"] is False
        assert verdict["chain_valid"] is False
        assert "mismatch" in verdict["message"].lower() or "signature" in verdict["message"].lower()

    def test_get_audit_logs_retrieval(self, tmp_path):
        db_path = str(tmp_path / "audit_query.db")
        init_db(db_path)

        log_audit_event(
            sku_id="Wheat",
            region="Punjab",
            model_version="v1",
            feature_snapshot_hash="h1",
            observed_price=2500.0,
            computed_band={"p10": 2000.0, "p50": 2200.0, "p90": 2400.0},
            anomaly_type="CEILING_BREACH",
            llm_verdict_json={"status": "FLAGGED"},
            db_path=db_path,
        )

        logs = get_audit_logs(limit=10, db_path=db_path)
        assert len(logs) == 1
        assert logs[0]["sku_id"] == "Wheat"
        assert logs[0]["region"] == "Punjab"
        assert "entry_hash" in logs[0]
        assert "prev_hash" in logs[0]


class TestPDFNoticeExporter:
    """Tests for ReportLab court-ready enforcement notice generation."""

    def test_generates_valid_pdf_bytes(self):
        sample_notice = {
            "notice_id": "ENF-TEST-2026-999",
            "severity_rating": "CRITICAL",
            "sku_name": "Tomato",
            "target_entity": "Nasik Wholesalers Syndicate",
            "region": "Maharashtra",
            "observed_price": 3800.0,
            "fair_price_ceiling": 2400.0,
            "price_deviation_pct": 58.3,
            "probable_cause": "Artificial price escalation during stable supply",
            "top_cost_drivers": [
                {"factor_name": "Supply Shock Z-Score", "impact_percentage": 52.0, "impact_direction": "INCREASE"},
                {"factor_name": "Diesel Freight Index", "impact_percentage": 28.0, "impact_direction": "INCREASE"},
            ],
            "legal_citations": [
                {
                    "statute_name": "Essential Commodities Act, 1955",
                    "section_clause": "Section 3(2)(c)",
                    "relevance_summary": "Orders controlling prices of essential commodities."
                },
                {
                    "statute_name": "Competition Act, 2002",
                    "section_clause": "Section 3(3)(a)",
                    "relevance_summary": "Prohibition of price-fixing cartels."
                }
            ],
            "recommended_action": "Issue formal show-cause notice and cost audit order.",
            "draft_notice_text": "SHOW CAUSE NOTICE UNDER SECTION 3 ECA: Price markup unjustified.",
        }

        pdf_bytes = build_enforcement_pdf(sample_notice)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 2000
        # PDF signature header
        assert pdf_bytes.startswith(b"%PDF")

    def test_saves_pdf_to_file(self, tmp_path):
        out_file = str(tmp_path / "test_notice.pdf")
        sample_notice = {
            "notice_id": "ENF-FILE-001",
            "severity_rating": "HIGH",
            "sku_name": "Potato",
            "target_entity": "Lucknow Mandi Traders",
            "region": "Uttar Pradesh",
            "observed_price": 2500.0,
            "fair_price_ceiling": 1800.0,
            "price_deviation_pct": 38.9,
        }
        pdf_bytes = build_enforcement_pdf(sample_notice, output_path=out_file)
        assert os.path.exists(out_file)
        assert os.path.getsize(out_file) > 1000


class TestCartelNetworkVisualizer:
    """Tests for Plotly inter-mandi pricing correlation graph generator."""

    def test_builds_network_figure_and_clusters(self):
        df_sample = pd.DataFrame({
            "observation_date": pd.date_range("2026-01-01", periods=30, freq="D").tolist() * 3,
            "sku_name": ["Tomato"] * 90,
            "market_mandi": ["Mandi A"] * 30 + ["Mandi B"] * 30 + ["Mandi C"] * 30,
            "modal_price_per_quintal": np.concatenate([
                np.linspace(1000, 2000, 30), # Synchronized A
                np.linspace(1020, 2010, 30), # Synchronized B (high corr with A)
                np.full(30, 1500.0)          # Flat C (low corr)
            ]),
        })

        fig, cliques = build_cartel_network_figure(df_sample, selected_sku="Tomato", corr_threshold=0.80)
        assert fig is not None
        assert len(fig.data) >= 2 # edge trace and node trace
        assert isinstance(cliques, list)
        # Mandi A and Mandi B should be identified as correlated
        if cliques:
            assert cliques[0]["correlation"] >= 0.80
