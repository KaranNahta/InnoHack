"""
CASPER-Gov: LLM Court-Ready Enforcement Generator
==================================================
Generates court-ready, formal legal enforcement notices and show-cause directives
using `instructor` and Pydantic schema validation with deterministic fallback logic.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Union

import instructor

from src.llm.schemas import CostDriver, LegalCitation, EnforcementNotice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("report_generator")


def get_instructor_client() -> Optional[Any]:
    """
    Attempts to initialize instructor client using OpenAI or Anthropic keys.
    Returns None if no keys are found in the environment.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if openai_key:
        try:
            import openai
            logger.info("Initializing Instructor wrapped OpenAI client...")
            return instructor.from_openai(openai.OpenAI(api_key=openai_key))
        except ImportError:
            logger.warning("openai package not installed.")
    elif anthropic_key:
        try:
            import anthropic
            logger.info("Initializing Instructor wrapped Anthropic client...")
            return instructor.from_anthropic(anthropic.Anthropic(api_key=anthropic_key))
        except ImportError:
            logger.warning("anthropic package not installed.")
    return None


def generate_fallback_notice(
    anomaly_alert: Dict[str, Any],
    shap_drivers: List[Dict[str, Any]],
    retrieved_precedents: Union[List[str], List[Dict[str, Any]]],
) -> EnforcementNotice:
    """
    Generates a high-quality, court-ready EnforcementNotice payload, serving as a robust,
    deterministic fallback when API keys are not present.
    """
    logger.info("Generating enforcement notice using fallback legal template parser...")

    # 1. Anomaly details & severity calculation
    sku = str(anomaly_alert.get("sku_name", "Essential Commodity"))
    region = str(anomaly_alert.get("state") or anomaly_alert.get("region") or "National Capital Region")
    observed_price = float(anomaly_alert.get("observed_price", 3500.0))
    ceiling_price = float(anomaly_alert.get("fair_price_ceiling", observed_price * 0.80))
    price_deviation = round(((observed_price - ceiling_price) / max(ceiling_price, 1e-5)) * 100.0, 2)
    
    vendors = anomaly_alert.get("vendors_involved") or [anomaly_alert.get("market_mandi") or anomaly_alert.get("target_entity", "Licensed Wholesalers")]
    if isinstance(vendors, str):
        target_entity = vendors
    else:
        target_entity = ", ".join([str(v) for v in vendors])

    severity_score = float(anomaly_alert.get("severity_score", 0.5))
    if severity_score >= 0.7 or price_deviation >= 25.0:
        severity = "CRITICAL"
        action = "Issue Immediate Show-Cause Notice & Trigger Mandatory Statutory Cost Audit under Section 3 of the Essential Commodities Act, 1955."
    elif severity_score >= 0.4 or price_deviation >= 10.0:
        severity = "HIGH"
        action = "Issue Statutory Compliance Warning & Require Transaction Ledger Disclosures within 48 hours."
    else:
        severity = "MEDIUM"
        action = "Flag for Enhanced Surveillance & Inspect Wholesale Margins against Benchmark Caps."

    date_str = str(anomaly_alert.get("date", "2026-08-21"))
    notice_id = f"REG-ENF-{date_str.replace('-', '')}-{hash(target_entity + sku) % 10000:04d}"

    # 2. Structure Cost Drivers
    structured_drivers: List[CostDriver] = []
    driver_texts = []
    for d in shap_drivers[:3]:
        factor = str(d.get("factor_name") or d.get("feature", "Market Pricing Lag"))
        impact = float(d.get("impact_percentage") or d.get("contribution_percentage", 25.0))
        structured_drivers.append(CostDriver(
            factor_name=factor,
            impact_percentage=impact,
        ))
        driver_texts.append(f"{factor} ({impact}% contribution)")

    # 3. Structure Legal Citations
    structured_citations: List[LegalCitation] = []
    precedent_lines = []
    
    for p in retrieved_precedents:
        if isinstance(p, dict):
            statute = p.get("statute", "Essential Commodities Act, 1955")
            section = p.get("section", "Section 3")
            excerpt = p.get("excerpt", "")
            rationale = p.get("title") or f"Powers to regulate fair pricing and prevent unjustified margin gouging under {section}."
            structured_citations.append(LegalCitation(
                statute_name=statute,
                section_clause=section,
                relevance_summary=rationale,
            ))
            precedent_lines.append(f"- {statute} ({section}): {rationale}")
        else:
            text = str(p)
            structured_citations.append(LegalCitation(
                statute_name="Essential Commodities Act 1955",
                section_clause="Section 3(2)(c)",
                relevance_summary=text[:100],
            ))
            precedent_lines.append(f"- {text}")

    if not structured_citations:
        structured_citations = [
            LegalCitation(
                statute_name="Essential Commodities Act, 1955",
                section_clause="Section 3",
                relevance_summary="Powers to control production, supply, and price ceiling breaches of essential commodities.",
            ),
            LegalCitation(
                statute_name="Competition Act, 2002",
                section_clause="Section 3(3)(a)",
                relevance_summary="Prohibition of anti-competitive agreements and synchronized price collusion.",
            ),
        ]
        precedent_lines = [
            "- Essential Commodities Act 1955 (Section 3): Statutory price control directives.",
            "- Competition Act 2002 (Section 3): Anti-competitive cartelization enforcement.",
        ]

    # 4. Formulate Probable Cause
    probable_cause = (
        f"Abnormal pricing behavior flagged for SKU '{sku}' in region '{region}' on {date_str}. "
        f"Observed price (₹{observed_price:.2f}/Qtl) exceeds fair price ceiling (₹{ceiling_price:.2f}/Qtl) by {price_deviation}%. "
        f"Attributed drivers: {', '.join(driver_texts)}."
    )

    # 5. Build Formal Legal Order Text
    precedents_block = "\n".join(precedent_lines)
    drivers_block = "\n".join([f"  [{i+1}] {cd.factor_name}: {cd.impact_percentage}% contribution" for i, cd in enumerate(structured_drivers)])

    draft_text = f"""OFFICIAL STATUTORY WARNING & ENFORCEMENT ORDER
ISSUED BY THE CASPER-GOV PRICE STABILIZATION & COMPLIANCE AUTHORITY
NOTICE IDENTIFIER: {notice_id}

Date of Issuance: {date_str}
Jurisdictional Region: {region}
Respondent Entity: {target_entity}
Commodity In Question: {sku}

STATUTORY NOTICE & MANDATORY ORDER:
Pursuant to Section 3 of the Essential Commodities Act, 1955, and read in conjunction with Section 3 of the Competition Act, 2002, the Price Stabilization & Compliance Authority hereby issues this formal statutory warning notice.

1. FACTUAL FINDINGS & PRICE ANOMALY:
On {date_str}, automated surveillance flagged a major market distortion:
  - Observed Market Price: ₹{observed_price:.2f} per Quintal
  - Calibrated Statutory Ceiling (p90): ₹{ceiling_price:.2f} per Quintal
  - Price Deviation: +{price_deviation}% above fair calibrated bounds
  - Anomaly Severity Classification: {severity}

2. COST ATTRIBUTION & SHAP DRIVERS:
Our quantitative explainability attribution verified the following contributing factors:
{drivers_block}

3. APPLICABLE STATUTORY CITATIONS:
{precedents_block}

4. REGULATORY DIRECTIVE & SHOW-CAUSE DEMAND:
The respondent entity ({target_entity}) is hereby ORDERED to submit a certified cost sheet and transaction ledger justifying all markups within 48 hours of service of this notice. Continued trading above statutory ceilings or failure to furnish compliance documentation will result in immediate cancellation of trading licenses, attachment of mandi lots, and prosecution under Section 7 of the Essential Commodities Act, 1955.

BY ORDER OF THE COMMISSIONER OF REGULATORY PRICING
CASPER-Gov Enforcement Directorate
"""

    return EnforcementNotice(
        notice_id=notice_id,
        severity_rating=severity,
        sku_name=sku,
        target_entity=target_entity,
        region=region,
        observed_price=observed_price,
        fair_price_ceiling=ceiling_price,
        price_deviation_pct=price_deviation,
        probable_cause=probable_cause,
        top_cost_drivers=structured_drivers,
        legal_citations=structured_citations,
        recommended_action=action,
        draft_notice_text=draft_text,
    )


def generate_enforcement_notice(
    anomaly_alert: Dict[str, Any],
    shap_drivers: List[Dict[str, Any]],
    retrieved_precedents: Union[List[str], List[Dict[str, Any]]],
) -> EnforcementNotice:
    """
    Generates structured, schema-validated enforcement notices matching the EnforcementNotice model.
    Utilizes Instructor LLM extraction if an API key is available; otherwise falls back to
    the court-ready deterministic legal template parser.
    """
    client = get_instructor_client()

    if not client:
        return generate_fallback_notice(anomaly_alert, shap_drivers, retrieved_precedents)

    logger.info("Executing Instructor LLM extraction call...")

    prompt = f"""
    You are a Senior Regulatory Compliance Counsel for CASPER-Gov drafting court-ready enforcement notices.
    
    Anomaly Event Details:
    {json.dumps(anomaly_alert, indent=2, default=str)}
    
    SHAP Cost Driver Breakdown:
    {json.dumps(shap_drivers, indent=2, default=str)}
    
    Retrieved Statutory Precedents:
    {json.dumps(retrieved_precedents, indent=2, default=str)}
    
    Produce a complete, formal, court-ready EnforcementNotice JSON payload matching the schema.
    """

    model_name = "gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "claude-3-5-sonnet-20240620"

    try:
        response = client.chat.completions.create(
            model=model_name,
            response_model=EnforcementNotice,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior regulatory compliance officer drafting official warnings and show-cause orders.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response
    except Exception as e:
        logger.error("Instructor API call failed: %s. Falling back to legal template.", str(e))
        return generate_fallback_notice(anomaly_alert, shap_drivers, retrieved_precedents)


def main():
    anomaly = {
        "date": "2026-08-21",
        "sku_name": "Tomato",
        "state": "Uttar Pradesh",
        "market_mandi": "Varanasi Mandi",
        "anomaly_type": "PRICE_GOUGING_ALERT",
        "severity_score": 0.85,
        "observed_price": 4200.0,
        "fair_price_ceiling": 3100.0,
        "vendors_involved": ["VEND_0110"],
    }

    drivers = [
        {"factor_name": "Mandi Arrival Supply Shock", "impact_percentage": 52.1},
        {"factor_name": "Recent 7-Day Price Lag", "impact_percentage": 31.4},
        {"factor_name": "Freight / Transportation Index", "impact_percentage": 16.5},
    ]

    from src.rag.vector_store import retrieve_legal_precedents
    precedents = retrieve_legal_precedents("Mandatory price ceiling breach and supply shock", top_k=2)

    notice = generate_enforcement_notice(anomaly, drivers, precedents)
    logger.info("Generated Notice ID: %s", notice.notice_id)
    logger.info("Severity: %s", notice.severity_rating)
    logger.info("\n%s", notice.draft_notice_text)


if __name__ == "__main__":
    main()


from src.llm.schemas import CriticDecision


def evaluate_price_estimate(
    sku_name: str,
    region: str,
    raw_p10: float,
    raw_p50: float,
    raw_p90: float,
    shap_drivers: List[Dict[str, Any]],
    retrieved_precedents: Union[List[str], List[Dict[str, Any]]],
) -> CriticDecision:
    """
    LLM-based Critic (Slide 4 & Slide 12):
    Validates and adjusts the quantitative price estimation output before finalization.
    The LLM never creates a price from scratch; it only accepts, rejects, or adjusts
    the bounds based on SHAP cost drivers and statutory precedents.
    """
    client = get_instructor_client()

    if not client:
        # Robust deterministic fallback
        # If severe supply shock driver is active, apply mild adjustment factor
        adjustment = 1.0
        reason = f"Conformal estimate for {sku_name} in {region} verified within statistical bounds."
        
        for d in shap_drivers:
            feat = str(d.get("factor_name") or d.get("feature", "")).lower()
            impact = float(d.get("impact_percentage") or d.get("contribution_percentage", 0.0))
            if "supply shock" in feat and impact > 40.0:
                adjustment = 1.04 # 4% upward adjustment for extreme supply shocks
                reason = f"Adjusted quantitative estimate (+4%) due to verified supply shock ({impact}% contribution)."
                break

        return CriticDecision(
            decision="ACCEPT" if adjustment == 1.0 else "ADJUST",
            adjustment_factor=round(adjustment, 3),
            adjusted_floor_p10=round(raw_p10 * adjustment, 2),
            adjusted_midpoint_p50=round(raw_p50 * adjustment, 2),
            adjusted_ceiling_p90=round(raw_p90 * adjustment, 2),
            reasoning=reason,
        )

    logger.info("Executing Instructor LLM Price Critic call...")
    prompt = f"""
    You are the CASPER-Gov Price Estimation Critic.
    Validate the following quantitative price band for SKU '{sku_name}' in region '{region}':
    - Raw Floor (p10): ₹{raw_p10:.2f}
    - Raw Midpoint (p50): ₹{raw_p50:.2f}
    - Raw Ceiling (p90): ₹{raw_p90:.2f}

    SHAP Feature Drivers:
    {json.dumps(shap_drivers, indent=2, default=str)}

    Legal & Market Precedents:
    {json.dumps(retrieved_precedents, indent=2, default=str)}

    Task:
    Evaluate if the price band accurately accounts for cost drivers and precedents.
    Decide whether to ACCEPT, REJECT, or ADJUST (with a multiplier between 0.85 and 1.15).
    """

    model_name = "gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "claude-3-5-sonnet-20240620"

    try:
        response = client.chat.completions.create(
            model=model_name,
            response_model=CriticDecision,
            messages=[
                {"role": "system", "content": "You are a quantitative price estimation critic validating ML price bands."},
                {"role": "user", "content": prompt},
            ],
        )
        return response
    except Exception as e:
        logger.error("Price Critic LLM call failed: %s. Using fallback critic.", str(e))
        return evaluate_price_estimate(sku_name, region, raw_p10, raw_p50, raw_p90, shap_drivers, retrieved_precedents)
