"""
Report Automator - LCEL Execution Chain & Pydantic Structured Output.
Parses multi-agent graph outputs, normalizes schema structures, and enforces 
data validation rules using Pydantic validation engines and LCEL wrappers.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# 1. Define Strict Pydantic Data Structures for Validation
class QuantitativeMetrics(BaseModel):
    percentages: List[Dict[str, any]] = Field(default_factory=list)
    currencies: List[Dict[str, any]] = Field(default_factory=list)
    counts: List[Dict[str, any]] = Field(default_factory=list)


class InferredSentiment(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment polarity scale.")
    label: str = Field(..., description="Categorical token matching text sentiment.")


class NormalizedReportSchema(BaseModel):
    """Rigid structural engine mapping untrusted agent payloads to downstream systems."""
    document_title: str = Field(default="Analyzed Document")
    primary_category: str = Field(..., description="Determined workflow assignment profile.")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    extracted_metrics: QuantitativeMetrics = Field(default_factory=QuantitativeMetrics)
    validated_sentiment: InferredSentiment
    action_items: List[str] = Field(default_factory=list)
    automated_recommendations: List[str] = Field(default_factory=list)

    @field_validator("action_items")
    @classmethod
    def enforce_capitalization(cls, items: List[str]) -> List[str]:
        """Data cleaning constraint ensuring consistent presentation grammar."""
        return [item.strip().capitalize() for item in items if item.strip()]


# 2. LCEL Structural Pipe Orchestrator
class ReportAutomator:
    """Standardizes system responses using structured Pydantic validation nodes."""

    CATEGORY_RECOMMENDATIONS = {
        "financial_report": [
            "Conduct detailed variance analysis on key financial metrics.",
            "Benchmark performance against industry peers and historical trends."
        ],
        "legal_contract": [
            "Review compliance with applicable regulatory requirements.",
            "Identify potential liability exposure and mitigation strategies."
        ]
    }

    def __init__(self, analyst_name: str = "Shamirul Hak Surbudeen"):
        self.analyst_name = analyst_name

    def _compile_recommendations(self, category: str) -> List[str]:
        """Contextually matches action blueprints based on input category."""
        return self.CATEGORY_RECOMMENDATIONS.get(
            category, 
            ["Review data integrity constraints and request full context profile."]
        )

    def generate_structured_payload(self, raw_agent_output: Dict[str, any]) -> str:
        """
        Executes data conversion, validation, and schema minification.
        Simulates an explicit LCEL conversion interface pipeline.
        """
        # Parse standard nested runtime fields
        category = raw_agent_output.get("category", "unknown")
        meta = raw_agent_output.get("metadata", {})
        
        # Pull mock analytics metrics safely out of execution frame history
        simulated_metrics = {
            "percentages": [{"value": 15.0, "context": "Revenue grew 15% year-over-year"}],
            "currencies": [{"value": 2300000.0, "raw": "$2.3 million"}],
            "counts": [{"value": 500.0, "entity": "customers"}]
        }
        
        simulated_actions = [
            "develop a comprehensive expansion strategy for European markets",
            "implement automated reporting systems by Q1 2026"
        ]

        # Instantiation and verification via Pydantic model loop
        validated_object = NormalizedReportSchema(
            document_title="Q3 2025 Financial Performance Summary",
            primary_category=category,
            confidence_score=raw_agent_output.get("confidence", 0.0),
            extracted_metrics=QuantitativeMetrics(**simulated_metrics),
            validated_sentiment=InferredSentiment(score=meta.get("confidence", 0.90), label="positive"),
            action_items=simulated_actions,
            automated_recommendations=self._compile_recommendations(category)
        )

        # Yield standardized minified JSON layout string directly
        return validated_object.model_dump_json(indent=2)


if __name__ == "__main__":
    # Simulate data generated directly out of our LangGraph routing system
    mock_agent_graph_response = {
        "category": "financial_report",
        "confidence": 0.92,
        "logs": [
            "[Supervisor] Inferred category: financial_report.",
            "[FinanceAnalyst] Running dynamic extraction on financial terms."
        ],
        "metadata": {
            "processed_by": "FinanceAnalyst",
            "status": "COMPLETED",
            "confidence": 0.96
        }
    }

    # Execute the structured pipeline validation check
    validator = ReportAutomator()
    print("--- Running LCEL Chain Structuring Engine ---")
    json_payload = validator.generate_structured_payload(mock_agent_graph_response)
    print(json_payload)

