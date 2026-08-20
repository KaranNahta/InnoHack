"""
Pydantic schemas for CASPER-Gov structured LLM outputs and enforcement notices.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class CostDriver(BaseModel):
    factor_name: str = Field(description="Name of the cost/market factor, e.g., Transport Freight Index, Wholesale Shock")
    impact_percentage: float = Field(description="Percentage contribution to the price movement based on SHAP analysis")

    @property
    def feature(self) -> str:
        return self.factor_name

    @property
    def contribution_percentage(self) -> float:
        return self.impact_percentage


class LegalCitation(BaseModel):
    statute_name: str = Field(description="e.g., Essential Commodities Act 1955")
    section_clause: str = Field(description="e.g., Section 3(2)(c)")
    relevance_summary: str = Field(description="Brief legal rationale for invoking this statute")


class EnforcementNotice(BaseModel):
    notice_id: str = Field(description="Unique notice identifier, e.g., REG-ENF-2026-0815")
    severity_rating: str = Field(description="CRITICAL, HIGH, or MEDIUM")
    sku_name: str
    target_entity: str = Field(description="Name of the mandi, vendor, or cartel group")
    region: str
    observed_price: float
    fair_price_ceiling: float
    price_deviation_pct: float
    probable_cause: str = Field(description="Probable cause statement, e.g., Coordinated multi-vendor price collusion without cost justification")
    top_cost_drivers: List[CostDriver]
    legal_citations: List[LegalCitation]
    recommended_action: str = Field(description="e.g., Issue Show-Cause Notice / Inspection Order under Section 3 ECA")
    draft_notice_text: str = Field(description="Formal legal enforcement order written for submission to court or regulatory authority")

    @property
    def draft_enforcement_notice_text(self) -> str:
        return self.draft_notice_text
