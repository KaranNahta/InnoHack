import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import openai
import anthropic
import instructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("report_generator")

# Pydantic schema models
class CostDriver(BaseModel):
    feature: str = Field(..., description="Feature/indicator name")
    contribution_percentage: float = Field(..., description="SHAP feature contribution percentage")
    impact_direction: str = Field(..., description="INCREASE or DECREASE direction")

class EnforcementNotice(BaseModel):
    severity_rating: str = Field(..., description="Notice severity level: CRITICAL, HIGH, or MEDIUM")
    probable_cause: str = Field(..., description="Summarized probable cause of the pricing anomaly")
    top_cost_drivers: List[CostDriver] = Field(..., description="List of top cost drivers attributed by SHAP analysis")
    recommended_action: str = Field(..., description="Recommended regulatory action course")
    draft_enforcement_notice_text: str = Field(..., description="Formal legal warning prose of the warning notice")

def get_instructor_client() -> Optional[Any]:
    """
    Attempts to initialize instructor client using OpenAI or Anthropic keys.
    Returns None if no keys are found.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if openai_key:
        logger.info("Initializing Instructor wrapped OpenAI client...")
        return instructor.from_openai(openai.OpenAI(api_key=openai_key))
    elif anthropic_key:
        logger.info("Initializing Instructor wrapped Anthropic client...")
        return instructor.from_anthropic(anthropic.Anthropic(api_key=anthropic_key))
    return None

def generate_fallback_notice(
    anomaly_alert: Dict[str, Any],
    shap_drivers: List[Dict[str, Any]],
    retrieved_precedents: List[str]
) -> EnforcementNotice:
    """
    Generates a high-quality, court-ready EnforcementNotice payload, serving as a robust,
    deterministic fallback when API keys are not present.
    """
    logger.info("Generating enforcement notice using fallback template parser...")
    
    # 1. Determine severity
    severity_score = anomaly_alert.get("severity_score", 0.5)
    if severity_score >= 0.7:
        severity = "CRITICAL"
        action = "Issue immediate warning notice and trigger mandatory statutory cost audits under Section 3 of the Essential Commodities Act, 1955."
    elif severity_score >= 0.4:
        severity = "HIGH"
        action = "Issue compliance warning letter and require pricing disclosures from registered wholesalers within 48 hours."
    else:
        severity = "MEDIUM"
        action = "Flag for active monitoring and verify margins against statutory benchmark limits."
        
    # 2. Extract top cost drivers
    pydantic_drivers = []
    driver_texts = []
    for d in shap_drivers[:3]:
        pydantic_drivers.append(CostDriver(
            feature=d["feature"],
            contribution_percentage=d["contribution_percentage"],
            impact_direction=d["impact_direction"]
        ))
        dir_word = "upward pressure" if d["impact_direction"] == "INCREASE" else "downward pressure"
        driver_texts.append(f"{d['feature']} (contributing {d['contribution_percentage']}% {dir_word})")
        
    # 3. Probable cause formulation
    probable_cause = f"Abnormal pricing behavior flagged for SKU '{anomaly_alert.get('sku_name', 'Staple')}' in region '{anomaly_alert.get('state', 'Region')}' on {anomaly_alert.get('date', 'today')}. "
    probable_cause += "Drivers: " + ", ".join(driver_texts) + "."
    
    # 4. Format precedents
    precedents_block = "\n".join([f"- {doc}" for doc in retrieved_precedents])
    
    # 5. Build formal legal notice text
    notice_text = f"""OFFICIAL WARNING AND ENFORCEMENT NOTICE
ISSUED BY THE CASPER-GOV PRICE STABILIZATION & COMPLIANCE AUTHORITY

Date: {anomaly_alert.get('date', '2026-08-20')}
Region: {anomaly_alert.get('state', 'Punjab')}
To: Registered Wholesalers & Licensed Retailers (Vendors: {', '.join(anomaly_alert.get('vendors_involved', ['Licensed Wholesalers']))})
Subject: Mandatory Warning Notice of statutory Compliance Audit for {anomaly_alert.get('sku_name', 'Essential SKU')} pricing anomalies.

Pursuant to Section 3 of the Essential Commodities Act, 1955, you are hereby served a formal pricing warning notice.

On {anomaly_alert.get('date', '2026-08-20')}, the CASPER-Gov compliance monitoring platform flagged a major pricing anomaly regarding your transactions for SKU '{anomaly_alert.get('sku_name', 'Staple')}' within {anomaly_alert.get('state', 'Region')}:
  - Pricing Alert Type: {anomaly_alert.get('anomaly_type', 'PRICE_GOUGING_ALERT')}
  - Calculated Severity Rating: {severity}

COST ATTRIBUTION ANALYSIS:
Our SHAP attribution explainability models mapped the primary price drivers for this anomaly:
"""
    for idx, d in enumerate(shap_drivers[:3]):
        notice_text += f"  [{idx+1}] {d['feature']}: {d['contribution_percentage']}% contribution ({d['impact_direction']} impact)\n"
        
    notice_text += f"""
RELEVANT STATUTORY PRECEDENTS & REGULATORY GUIDELINES:
{precedents_block}

ORDER AND DIRECTIVES:
You are hereby commanded to submit a detailed transaction cost sheet ledger within 48 hours of receipt of this warning. Continued pricing of essential goods outside calibrated statutory bands, or failure to comply with margins audits, will lead to immediate cancellation of trading licenses, search and seizure of stock reserves, and prosecution under the Essential Commodities Act, 1955.

Issued by Order of the Price Stabilization Commissioner, CASPER-Gov.
"""
    return EnforcementNotice(
        severity_rating=severity,
        probable_cause=probable_cause,
        top_cost_drivers=pydantic_drivers,
        recommended_action=action,
        draft_enforcement_notice_text=notice_text
    )

def generate_enforcement_notice(
    anomaly_alert: Dict[str, Any],
    shap_drivers: List[Dict[str, Any]],
    retrieved_precedents: List[str]
) -> EnforcementNotice:
    """
    Generates structured, schema-validated enforcement notices.
    Utilizes instructor API call if key is set, else falls back to high-quality fallback template parser.
    """
    client = get_instructor_client()
    
    if not client:
        return generate_fallback_notice(anomaly_alert, shap_drivers, retrieved_precedents)
        
    logger.info("Executing Instructor LLM extraction call...")
    
    prompt = f"""
    You are a regulatory system notice writer for CASPER-Gov. Write a court-ready warning notice.
    
    Anomaly Details:
    {json.dumps(anomaly_alert, indent=2)}
    
    SHAP Cost Driver Breakdown:
    {json.dumps(shap_drivers, indent=2)}
    
    Retrieved Legal Precedents:
    {chr(10).join(retrieved_precedents)}
    
    Extract and structure the legal enforcement notice payload matching the EnforcementNotice schema.
    """
    
    # Select LLM model based on key
    if os.environ.get("OPENAI_API_KEY"):
        model_name = "gpt-4o-mini"
    else:
        model_name = "claude-3-5-sonnet-20240620"
        
    try:
        response = client.chat.completions.create(
            model=model_name,
            response_model=EnforcementNotice,
            messages=[
                {"role": "system", "content": "You are a senior regulatory compliance officer drafting official warnings."},
                {"role": "user", "content": prompt}
            ]
        )
        return response
    except Exception as e:
        logger.error("Instructor API call failed: %s. Falling back to template.", str(e))
        return generate_fallback_notice(anomaly_alert, shap_drivers, retrieved_precedents)

def main():
    # Simple CLI test
    import json
    anomaly = {
        "date": "2026-08-20",
        "sku_name": "Tomato",
        "state": "Uttar Pradesh",
        "anomaly_type": "PRICE_GOUGING_ALERT",
        "severity_score": 0.85,
        "vendors_involved": ["VEND_0110"]
    }
    
    drivers = [
        {"feature": "Mandi Arrival Supply Shock", "contribution_percentage": 52.1, "impact_direction": "INCREASE"},
        {"feature": "Recent 7-Day Price Lag", "contribution_percentage": 31.4, "impact_direction": "INCREASE"},
        {"feature": "Freight/Transportation Cost Index", "contribution_percentage": 16.5, "impact_direction": "INCREASE"}
    ]
    
    precedents = [
        "Essential Commodities Act (Section 3) - Section 3 allows the government to intervene and penalize artificial shortages.",
        "Regulatory Intervention Policy - Modal price ceiling breach triggers warnings."
    ]
    
    notice = generate_enforcement_notice(anomaly, drivers, precedents)
    logger.info("Generated Enforcement Notice payload:")
    logger.info("\n%s", notice.model_dump_json(indent=4))

if __name__ == "__main__":
    main()
