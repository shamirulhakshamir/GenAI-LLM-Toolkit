"""Tests for InsightExtractor."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.insight_extractor import InsightExtractor


@pytest.fixture
def extractor():
    return InsightExtractor()


class TestExtractMetrics:
    def test_extract_percentages(self, extractor):
        metrics = extractor.extract_metrics("Revenue grew 15% year-over-year")
        assert len(metrics["percentages"]) == 1
        assert metrics["percentages"][0]["value"] == 15.0

    def test_extract_multiple_percentages(self, extractor):
        metrics = extractor.extract_metrics("Margin improved from 18% to 22.5%")
        assert len(metrics["percentages"]) == 2

    def test_extract_decimal_percentage(self, extractor):
        metrics = extractor.extract_metrics("Growth rate of 3.7%")
        assert metrics["percentages"][0]["value"] == 3.7

    def test_extract_dollar_currency(self, extractor):
        metrics = extractor.extract_metrics("Revenue reached $2.3 million")
        assert len(metrics["currencies"]) == 1
        assert metrics["currencies"][0]["value"] == 2_300_000

    def test_extract_pound_currency(self, extractor):
        metrics = extractor.extract_metrics("Profit of \u00a3800K")
        assert len(metrics["currencies"]) == 1
        assert metrics["currencies"][0]["value"] == 800_000

    def test_extract_euro_currency(self, extractor):
        metrics = extractor.extract_metrics("Budget of \u20ac500 thousand")
        assert len(metrics["currencies"]) == 1
        assert metrics["currencies"][0]["value"] == 500_000

    def test_extract_billion_currency(self, extractor):
        metrics = extractor.extract_metrics("Total assets $4.2 billion")
        assert metrics["currencies"][0]["value"] == 4_200_000_000

    def test_extract_counts(self, extractor):
        metrics = extractor.extract_metrics("Added 500 customers during Q3")
        assert len(metrics["counts"]) == 1
        assert metrics["counts"][0]["value"] == 500
        assert metrics["counts"][0]["entity"] == "customers"

    def test_extract_comma_separated_counts(self, extractor):
        metrics = extractor.extract_metrics("Total of 12,000 users")
        assert metrics["counts"][0]["value"] == 12000

    def test_no_metrics_in_plain_text(self, extractor):
        metrics = extractor.extract_metrics("The company performed well this quarter")
        assert len(metrics["percentages"]) == 0
        assert len(metrics["currencies"]) == 0
        assert len(metrics["counts"]) == 0


class TestAnalyzeSentiment:
    def test_positive_sentiment(self, extractor):
        result = extractor.analyze_sentiment(
            "Revenue increased strongly and growth exceeded expectations"
        )
        assert result["label"] == "positive"
        assert result["score"] > 0

    def test_negative_sentiment(self, extractor):
        result = extractor.analyze_sentiment(
            "Performance declined and losses increased with weak outlook"
        )
        assert result["label"] == "negative"
        assert result["score"] < 0

    def test_neutral_sentiment(self, extractor):
        result = extractor.analyze_sentiment(
            "The document contains standard operating procedures"
        )
        assert result["label"] == "neutral"

    def test_sentiment_returns_indicators(self, extractor):
        result = extractor.analyze_sentiment("Revenue grew and profit improved significantly")
        assert "grew" in result["positive_indicators"]
        assert "improved" in result["positive_indicators"]

    def test_sentiment_score_in_range(self, extractor):
        result = extractor.analyze_sentiment("A mixed bag of growth and decline")
        assert -1.0 <= result["score"] <= 1.0


class TestExtractActionItems:
    def test_must_pattern(self, extractor):
        actions = extractor.extract_action_items(
            "Management must develop a comprehensive expansion strategy."
        )
        assert len(actions) >= 1
        assert any("expansion strategy" in a.lower() for a in actions)

    def test_should_pattern(self, extractor):
        actions = extractor.extract_action_items(
            "The team should implement automated reporting by Q1."
        )
        assert len(actions) >= 1

    def test_need_to_pattern(self, extractor):
        actions = extractor.extract_action_items(
            "We need to review the pricing structures for next quarter."
        )
        assert len(actions) >= 1

    def test_ensure_pattern(self, extractor):
        actions = extractor.extract_action_items(
            "Ensure compliance with all regulatory requirements."
        )
        assert len(actions) >= 1

    def test_no_actions_in_descriptive_text(self, extractor):
        actions = extractor.extract_action_items(
            "The company reported quarterly earnings of two dollars per share."
        )
        # Should have no or very few action items
        assert len(actions) <= 1


class TestExtractDates:
    def test_quarter_year(self, extractor):
        dates = extractor.extract_dates("Results for Q3 2025 were strong")
        assert "Q3 2025" in dates

    def test_fiscal_year(self, extractor):
        dates = extractor.extract_dates("Guidance for FY2026 raised")
        assert "FY2026" in dates

    def test_month_year(self, extractor):
        dates = extractor.extract_dates("Starting January 2025")
        assert "January 2025" in dates

    def test_multiple_dates(self, extractor):
        dates = extractor.extract_dates("From Q1 2025 through Q4 2025")
        assert len(dates) >= 2


class TestGetKeyTerms:
    def test_returns_list_of_tuples(self, extractor):
        terms = extractor.get_key_terms("Revenue growth and market expansion strategy", top_n=3)
        assert isinstance(terms, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in terms)

    def test_excludes_stopwords(self, extractor):
        terms = extractor.get_key_terms("The revenue is growing in the market")
        term_words = [t[0] for t in terms]
        assert "the" not in term_words
        assert "is" not in term_words

    def test_top_n_limit(self, extractor):
        text = "Revenue growth market expansion customers strategy analytics data pipeline"
        terms = extractor.get_key_terms(text, top_n=3)
        assert len(terms) <= 3


class TestFullExtraction:
    def test_extract_returns_all_keys(self, extractor):
        result = extractor.extract("Revenue grew 15% to $2.3 million with 500 customers")
        assert "metrics" in result
        assert "sentiment" in result
        assert "action_items" in result
        assert "key_dates" in result
        assert "word_frequency" in result
        assert "summary_stats" in result

    def test_summary_stats_keys(self, extractor):
        result = extractor.extract("This is a test document. It has two sentences.")
        stats = result["summary_stats"]
        assert "word_count" in stats
        assert "sentence_count" in stats
        assert "avg_sentence_length" in stats
        assert "character_count" in stats

    def test_summary_stats_values(self, extractor):
        result = extractor.extract("Hello world. Testing one two three.")
        stats = result["summary_stats"]
        assert stats["word_count"] > 0
        assert stats["sentence_count"] >= 2
        assert stats["character_count"] > 0

    def test_empty_text(self, extractor):
        result = extractor.extract("")
        assert result["metrics"]["percentages"] == []
        assert result["metrics"]["currencies"] == []
        assert result["summary_stats"]["word_count"] == 0
