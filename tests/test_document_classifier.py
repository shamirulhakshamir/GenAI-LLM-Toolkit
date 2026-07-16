"""Tests for DocumentClassifier (LangGraph supervisor/worker orchestration)."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.document_classifier import (
    DocumentClassifier,
    DocumentOrchestrator,
    route_to_specialist,
)


@pytest.fixture(scope="module")
def classifier():
    """Build the compiled LangGraph pipeline once for all tests."""
    return DocumentClassifier()


class TestDocumentClassifierInit:
    def test_init_builds_orchestrator_workers_and_graph(self, classifier):
        assert isinstance(classifier.orchestrator, DocumentOrchestrator)
        assert classifier.workers is not None
        assert classifier.graph is not None


class TestDocumentOrchestratorIntentRouting:
    """Unit-level checks on the supervisor's intent classification, no graph involved."""

    @pytest.mark.parametrize(
        "text,expected_intent,expected_analyst",
        [
            ("Revenue grew 15% and EBITDA margin improved", "financial_report", "FinanceAnalystAgent"),
            ("This agreement shall be governed by the laws of Ireland", "legal_contract", "LegalComplianceAgent"),
            ("All employees are entitled to paid leave under HR policy", "hr_policy", "HRPolicyAgent"),
            ("The microservices architecture exposes a REST API over a database", "technical_spec", "ArchitectureAgent"),
            ("Our new campaign targets younger audiences", "marketing_brief", "BrandStrategistAgent"),
        ],
    )
    def test_analyze_intent_classification(self, text, expected_intent, expected_analyst):
        orchestrator = DocumentOrchestrator()
        state = {
            "document_text": text,
            "inferred_intent": "",
            "assigned_analyst": "",
            "validation_passed": True,
            "final_payload": {},
            "execution_logs": [],
        }
        result = orchestrator.analyze_intent(state)
        assert result["inferred_intent"] == expected_intent
        assert result["assigned_analyst"] == expected_analyst
        assert len(result["execution_logs"]) == 2


class TestRouteToSpecialist:
    def test_routes_financial_report_to_finance(self):
        assert route_to_specialist({"inferred_intent": "financial_report"}) == "finance"

    def test_routes_legal_contract_to_legal(self):
        assert route_to_specialist({"inferred_intent": "legal_contract"}) == "legal"

    @pytest.mark.parametrize("intent", ["hr_policy", "technical_spec", "marketing_brief"])
    def test_routes_everything_else_to_general(self, intent):
        assert route_to_specialist({"inferred_intent": intent}) == "general"


class TestDocumentClassifierWorkflow:
    def test_run_workflow_returns_expected_keys(self, classifier):
        result = classifier.run_workflow("Q3 revenue increased by 12%")
        assert "category" in result
        assert "confidence" in result
        assert "logs" in result
        assert "metadata" in result

    def test_financial_document_routes_to_finance_worker(self, classifier):
        result = classifier.run_workflow(
            "EBITDA reached $156 million representing a 15% improvement over prior year"
        )
        assert result["category"] == "financial_report"
        assert result["metadata"]["processed_by"] == "FinanceAnalyst"
        assert result["confidence"] == 0.96

    def test_legal_document_routes_to_legal_worker(self, classifier):
        result = classifier.run_workflow(
            "This agreement shall be governed by the laws of the Netherlands"
        )
        assert result["category"] == "legal_contract"
        assert result["metadata"]["processed_by"] == "LegalCompliance"
        assert result["confidence"] == 0.94

    def test_hr_document_routes_to_general_worker(self, classifier):
        result = classifier.run_workflow(
            "All employees are entitled to 25 days of annual leave per year"
        )
        assert result["category"] == "hr_policy"
        assert result["metadata"]["processed_by"] == "HRPolicyAgent"
        assert result["confidence"] == 0.89

    def test_technical_document_routes_to_general_worker(self, classifier):
        result = classifier.run_workflow(
            "The microservices architecture uses Kubernetes with auto-scaling"
        )
        assert result["category"] == "technical_spec"
        assert result["metadata"]["processed_by"] == "ArchitectureAgent"

    def test_marketing_document_falls_back_to_general(self, classifier):
        result = classifier.run_workflow(
            "The campaign targets C-suite decision makers with social media strategy"
        )
        assert result["category"] == "marketing_brief"
        assert result["metadata"]["processed_by"] == "BrandStrategistAgent"

    def test_confidence_in_valid_range(self, classifier):
        result = classifier.run_workflow("Revenue grew 15% year-over-year")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_execution_logs_capture_supervisor_and_worker_steps(self, classifier):
        result = classifier.run_workflow("Revenue grew 15% year-over-year")
        assert any("[Supervisor]" in line for line in result["logs"])
        assert any("[FinanceAnalyst]" in line for line in result["logs"])

    def test_each_call_starts_with_fresh_log_state(self, classifier):
        """Ensures state isn't leaking/accumulating across independent workflow runs."""
        first = classifier.run_workflow("Revenue grew 15% year-over-year")
        second = classifier.run_workflow("This agreement is governed by Irish law")
        assert first["logs"] != second["logs"]
        assert len(second["logs"]) == 3
