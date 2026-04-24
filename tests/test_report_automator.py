"""Tests for ReportAutomator."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.report_automator import ReportAutomator


@pytest.fixture
def automator():
    return ReportAutomator()


@pytest.fixture
def sample_classification():
    return {
        "category": "financial_report",
        "confidence": 0.92,
        "all_scores": {
            "financial_report": 0.92,
            "legal_contract": 0.03,
            "hr_policy": 0.01,
            "technical_spec": 0.02,
            "marketing_brief": 0.02,
        },
    }


@pytest.fixture
def sample_insights():
    return {
        "metrics": {
            "percentages": [
                {"value": 15.0, "context": "Revenue grew 15% year-over-year"},
            ],
            "currencies": [
                {"value": 2_300_000, "raw": "$2.3 million", "context": "revenue to $2.3 million"},
            ],
            "counts": [
                {"value": 500, "entity": "customers"},
            ],
        },
        "sentiment": {
            "score": 0.6,
            "label": "positive",
            "positive_indicators": ["grew", "improved"],
            "negative_indicators": [],
        },
        "action_items": [
            "develop a comprehensive expansion strategy",
            "implement automated reporting systems",
        ],
        "key_dates": ["Q3 2025"],
        "word_frequency": [
            ("revenue", 3), ("market", 2), ("growth", 2),
        ],
        "summary_stats": {
            "word_count": 100,
            "sentence_count": 6,
            "avg_sentence_length": 16.7,
            "character_count": 550,
        },
    }


class TestReportAutomatorInit:
    def test_default_init(self):
        automator = ReportAutomator()
        assert automator.analyst_name == "Shamirul Hak Surbudeen"
        assert automator.firm_name == "EY"

    def test_custom_init(self):
        automator = ReportAutomator(analyst_name="Test User", firm_name="TestCo")
        assert automator.analyst_name == "Test User"
        assert automator.firm_name == "TestCo"


class TestGenerateReport:
    def test_report_is_string(self, automator, sample_classification, sample_insights):
        report = automator.generate(sample_classification, sample_insights)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_report_contains_header(self, automator, sample_classification, sample_insights):
        report = automator.generate(sample_classification, sample_insights)
        assert "CONSULTING ANALYSIS REPORT" in report
        assert "EY" in report

    def test_report_contains_executive_summary(self, automator, sample_classification, sample_insights):
        report = automator.generate(sample_classification, sample_insights)
        assert "EXECUTIVE SUMMARY" in report

    def test_report_contains_classification(self, automator, sample_classification, sample_insights):
        report = automator.generate(sample_classification, sample_insights)
        assert "DOCUMENT CLASSIFICATION" in report
        assert "Financial Report" in report

    def test_report_contains_key_findings(self, automator, sample_classification, sample_insights):
        report = automator.generate(sample_classification, sample_insights)
        assert "KEY FINDINGS" in report

    def test_report_contains_metrics(self, automator, sample_classification, sample_insights):
        report = automator.generate(sample_classification, sample_insights)
        assert "QUANTITATIVE METRICS" in report
        assert "15.0%" in report

    def test_report_contains_sentiment(self, automator, sample_classification, sample_insights):
        report = automator.generate(sample_classification, sample_insights)
        assert "SENTIMENT ANALYSIS" in report
        assert "POSITIVE" in report

    def test_report_contains_action_items(self, automator, sample_classification, sample_insights):
        report = automator.generate(sample_classification, sample_insights)
        assert "ACTION ITEMS" in report
        assert "expansion strategy" in report

    def test_report_contains_recommendations(self, automator, sample_classification, sample_insights):
        report = automator.generate(sample_classification, sample_insights)
        assert "RECOMMENDATIONS" in report

    def test_report_contains_footer(self, automator, sample_classification, sample_insights):
        report = automator.generate(sample_classification, sample_insights)
        assert "CONFIDENTIAL" in report
        assert "Shamirul Hak Surbudeen" in report

    def test_custom_document_title(self, automator, sample_classification, sample_insights):
        report = automator.generate(
            sample_classification, sample_insights,
            document_title="Q3 2025 Performance Review"
        )
        assert "Q3 2025 Performance Review" in report


class TestGenerateSummary:
    def test_summary_is_string(self, automator, sample_classification, sample_insights):
        summary = automator.generate_summary(sample_classification, sample_insights)
        assert isinstance(summary, str)
        assert len(summary) > 20

    def test_summary_contains_category(self, automator, sample_classification, sample_insights):
        summary = automator.generate_summary(sample_classification, sample_insights)
        assert "Financial Report" in summary

    def test_summary_contains_sentiment(self, automator, sample_classification, sample_insights):
        summary = automator.generate_summary(sample_classification, sample_insights)
        assert "positive" in summary.lower()

    def test_summary_contains_metric_count(self, automator, sample_classification, sample_insights):
        summary = automator.generate_summary(sample_classification, sample_insights)
        # Should mention 3 total metrics (1 pct + 1 currency + 1 count)
        assert "3" in summary


class TestAllCategories:
    @pytest.mark.parametrize("category,label", [
        ("financial_report", "Financial Report"),
        ("legal_contract", "Legal Contract"),
        ("hr_policy", "HR Policy Document"),
        ("technical_spec", "Technical Specification"),
        ("marketing_brief", "Marketing Brief"),
    ])
    def test_category_recommendations_exist(self, automator, category, label):
        classification = {"category": category, "confidence": 0.9, "all_scores": {category: 0.9}}
        insights = {
            "metrics": {"percentages": [], "currencies": [], "counts": []},
            "sentiment": {"score": 0, "label": "neutral", "positive_indicators": [], "negative_indicators": []},
            "action_items": [],
            "key_dates": [],
            "word_frequency": [],
            "summary_stats": {"word_count": 0, "sentence_count": 0, "avg_sentence_length": 0, "character_count": 0},
        }
        report = automator.generate(classification, insights)
        assert label in report
        assert "RECOMMENDATIONS" in report


class TestEdgeCases:
    def test_empty_insights(self, automator):
        classification = {"category": "financial_report", "confidence": 0.5, "all_scores": {}}
        insights = {
            "metrics": {"percentages": [], "currencies": [], "counts": []},
            "sentiment": {"score": 0, "label": "neutral", "positive_indicators": [], "negative_indicators": []},
            "action_items": [],
            "key_dates": [],
            "word_frequency": [],
            "summary_stats": {"word_count": 0, "sentence_count": 0, "avg_sentence_length": 0, "character_count": 0},
        }
        report = automator.generate(classification, insights)
        assert isinstance(report, str)
        assert "No quantitative metrics found" in report
        assert "No action items identified" in report

    def test_unknown_category(self, automator):
        classification = {"category": "unknown_type", "confidence": 0.3, "all_scores": {}}
        insights = {
            "metrics": {"percentages": [], "currencies": [], "counts": []},
            "sentiment": {"score": 0, "label": "neutral", "positive_indicators": [], "negative_indicators": []},
            "action_items": [],
            "key_dates": [],
            "word_frequency": [],
            "summary_stats": {"word_count": 0, "sentence_count": 0, "avg_sentence_length": 0, "character_count": 0},
        }
        report = automator.generate(classification, insights)
        assert isinstance(report, str)
        assert "No category-specific recommendations available" in report
