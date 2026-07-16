"""Tests for LCEL and Pydantic-enforced ReportAutomator."""

import sys
import os
import json
import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.report_automator import ReportAutomator, NormalizedReportSchema


@pytest.fixture
def validator():
    """Instantiate a clean ReportAutomator component instance for validation."""
    return ReportAutomator()


@pytest.fixture
def mock_graph_response():
    """Simulates a valid dictionary payload emitted from an upstream LangGraph workflow."""
    return {
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


class TestReportAutomatorPydanticSchema:
    def test_successful_validation_and_serialization(self, validator, mock_graph_response):
        """Ensure valid agent parameters serialize cleanly into structured minified JSON."""
        json_string = validator.generate_structured_payload(mock_graph_response)
        
        # Verify result parses back as valid JSON
        parsed_data = json.loads(json_string)
        assert parsed_data["primary_category"] == "financial_report"
        assert parsed_data["confidence_score"] == 0.92
        assert "validated_sentiment" in parsed_data
        assert isinstance(parsed_data["action_items"], list)

    def test_data_cleaning_validator_rule(self, validator, mock_graph_response):
        """Ensure the Pydantic field validator cleanly strips and capitalizes items."""
        json_string = validator.generate_structured_payload(mock_graph_response)
        parsed_data = json.loads(json_string)
        
        # Check that action items were automatically normalized by the schema rule
        for item in parsed_data["action_items"]:
            assert item[0].isupper()
            assert not item.startswith(" ")

    def test_schema_rejections_on_invalid_bounds(self, validator):
        """Ensure validation engines throw exceptions when state variables break contract rules."""
        # Confidence scores must be less than or equal to 1.0
        malformed_response = {
            "category": "legal_contract",
            "confidence": 99.9,  # Fatal boundary error
            "metadata": {}
        }
        
        with pytest.raises(ValidationError):
            validator.generate_structured_payload(malformed_response)


class TestReportAutomatorLCELLogic:
    def test_compiled_recommendations_matching(self, validator, mock_graph_response):
        """Verify business domain blueprints are correctly mapped based on structural keys."""
        json_string = validator.generate_structured_payload(mock_graph_response)
        parsed_data = json.loads(json_string)
        
        recs = parsed_data["automated_recommendations"]
        assert len(recs) > 0
        assert "variance analysis" in recs[0].lower()

    def test_fallback_recommendations_on_unknown_category(self, validator):
        """Ensure standard defensive blueprints populate safely if categorization fails."""
        unmapped_response = {
            "category": "unsupported_legacy_format",
            "confidence": 0.45,
            "metadata": {"confidence": 0.50}
        }
        
        json_string = validator.generate_structured_payload(unmapped_response)
        parsed_data = json.loads(json_string)
        
        recs = parsed_data["automated_recommendations"]
        assert "data integrity" in recs[0].lower()
