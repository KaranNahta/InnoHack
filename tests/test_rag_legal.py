"""
Unit tests for RAG Legal Vector Store (src/rag/vector_store.py)
and LLM Schemas (src/llm/schemas.py).
"""

import os
import pytest
from pydantic import ValidationError

from src.rag.vector_store import (
    populate_database,
    retrieve_legal_precedents,
    query_precedents,
    STATUTORY_PRECEDENTS,
    COLLECTION_NAME,
)
from src.llm.schemas import CostDriver, LegalCitation, EnforcementNotice
from src.llm.report_generator import generate_enforcement_notice, generate_fallback_notice


def test_schemas_model_validation():
    # 1. CostDriver
    cd = CostDriver(factor_name="Diesel Transportation Index", impact_percentage=34.5)
    assert cd.factor_name == "Diesel Transportation Index"
    assert cd.impact_percentage == 34.5

    # 2. LegalCitation
    lc = LegalCitation(
        statute_name="Essential Commodities Act 1955",
        section_clause="Section 3(2)(c)",
        relevance_summary="Power to control maximum prices of food grains.",
    )
    assert lc.statute_name == "Essential Commodities Act 1955"
    assert "Section 3" in lc.section_clause

    # 3. EnforcementNotice
    notice = EnforcementNotice(
        notice_id="REG-ENF-2026-001",
        severity_rating="CRITICAL",
        sku_name="Onion",
        target_entity="Nasik Wholesalers Syndicate",
        region="Maharashtra",
        observed_price=4500.0,
        fair_price_ceiling=3200.0,
        price_deviation_pct=40.6,
        probable_cause="Coordinated price spike without freight or crop cost increase",
        top_cost_drivers=[cd],
        legal_citations=[lc],
        recommended_action="Issue immediate inspection order and search warrant",
        draft_notice_text="ORDER UNDER SECTION 3 ECA...",
    )
    assert notice.severity_rating == "CRITICAL"
    assert notice.price_deviation_pct == 40.6
    assert len(notice.top_cost_drivers) == 1
    assert len(notice.legal_citations) == 1


def test_vector_store_seeding_and_retrieval(tmp_path):
    db_path = str(tmp_path / "chromadb_test")

    # 1. Populate database
    populate_database(db_path=db_path)
    assert os.path.exists(os.path.join(db_path, "chroma.sqlite3"))

    # 2. Test retrieve_legal_precedents
    results = retrieve_legal_precedents("Section 3 Essential Commodities price fixing", top_k=2, db_path=db_path)
    assert isinstance(results, list)
    assert len(results) == 2

    # Verify structured dictionary return format
    top_res = results[0]
    assert "statute" in top_res
    assert "section" in top_res
    assert "excerpt" in top_res
    assert "relevance_score" in top_res
    assert isinstance(top_res["relevance_score"], float)
    assert 0.0 <= top_res["relevance_score"] <= 1.0

    # 3. Test antitrust cartel retrieval
    cartel_results = retrieve_legal_precedents("Synchronized vendor price collusion and cartelization", top_k=1, db_path=db_path)
    assert len(cartel_results) == 1
    assert "Competition Act" in cartel_results[0]["statute"] or "Essential Commodities" in cartel_results[0]["statute"]

    # 4. Test Legal Metrology retrieval
    mrp_results = retrieve_legal_precedents("Maximum Retail Price overcharging violations", top_k=1, db_path=db_path)
    assert len(mrp_results) == 1
    assert "Legal Metrology" in mrp_results[0]["statute"] or "MRP" in mrp_results[0]["excerpt"]


def test_enforcement_generator_fallback(tmp_path):
    db_path = str(tmp_path / "chromadb_enf")
    populate_database(db_path=db_path)
    
    precedents = retrieve_legal_precedents("Price gouging and artificial scarcity", top_k=2, db_path=db_path)
    
    anomaly = {
        "date": "2026-08-21",
        "sku_name": "Potato",
        "region": "Uttar Pradesh",
        "market_mandi": "Agra Mandi",
        "observed_price": 2800.0,
        "fair_price_ceiling": 1900.0,
        "severity_score": 0.88,
        "vendors_involved": ["VEND_0042", "VEND_0043"],
    }
    
    drivers = [
        {"factor_name": "Cold Storage Hoarding Index", "impact_percentage": 58.4},
        {"factor_name": "Wholesale Price Lag", "impact_percentage": 24.1},
    ]
    
    notice = generate_enforcement_notice(anomaly, drivers, precedents)
    
    assert isinstance(notice, EnforcementNotice)
    assert notice.severity_rating == "CRITICAL"
    assert notice.sku_name == "Potato"
    assert notice.region == "Uttar Pradesh"
    assert len(notice.top_cost_drivers) == 2
    assert len(notice.legal_citations) == 2
    assert "OFFICIAL STATUTORY WARNING" in notice.draft_notice_text
    assert "Agra Mandi" in notice.target_entity or "VEND_0042" in notice.target_entity
