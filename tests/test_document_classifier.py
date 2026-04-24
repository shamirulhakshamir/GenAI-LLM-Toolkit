"""Tests for DocumentClassifier."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.document_classifier import DocumentClassifier, SYNTHETIC_DOCUMENTS, CATEGORIES


@pytest.fixture(scope="module")
def trained_classifier():
    """Train classifier once for all tests."""
    clf = DocumentClassifier()
    clf.train()
    return clf


class TestDocumentClassifierInit:
    def test_init_creates_vectorizer_and_model(self):
        clf = DocumentClassifier()
        assert clf.vectorizer is not None
        assert clf.model is not None
        assert clf.is_trained is False

    def test_categories_match_synthetic_data(self):
        assert set(CATEGORIES) == set(SYNTHETIC_DOCUMENTS.keys())

    def test_synthetic_data_has_samples(self):
        for category, docs in SYNTHETIC_DOCUMENTS.items():
            assert len(docs) >= 5, f"{category} has too few samples"


class TestDocumentClassifierTraining:
    def test_train_returns_accuracy(self, trained_classifier):
        # Re-train to check return value
        clf = DocumentClassifier()
        results = clf.train()
        assert "accuracy" in results
        assert "report" in results
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_train_sets_is_trained(self, trained_classifier):
        assert trained_classifier.is_trained is True

    def test_accuracy_above_threshold(self):
        clf = DocumentClassifier()
        results = clf.train()
        # With synthetic data and logistic regression, expect reasonable accuracy
        assert results["accuracy"] >= 0.5, f"Accuracy too low: {results['accuracy']}"


class TestDocumentClassifierPrediction:
    def test_predict_returns_expected_keys(self, trained_classifier):
        result = trained_classifier.predict("Q3 revenue increased by 12%")
        assert "category" in result
        assert "confidence" in result
        assert "all_scores" in result

    def test_predict_category_is_valid(self, trained_classifier):
        result = trained_classifier.predict("The quarterly earnings show profit growth")
        assert result["category"] in CATEGORIES

    def test_predict_confidence_in_range(self, trained_classifier):
        result = trained_classifier.predict("Revenue grew 15% year-over-year")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_all_scores_sum_to_one(self, trained_classifier):
        result = trained_classifier.predict("Total assets grew to $4.2 billion")
        total = sum(result["all_scores"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_financial_document(self, trained_classifier):
        result = trained_classifier.predict(
            "EBITDA reached $156 million representing a 15% improvement over prior year"
        )
        assert result["category"] == "financial_report"

    def test_predict_legal_document(self, trained_classifier):
        result = trained_classifier.predict(
            "This agreement shall be governed by the laws of the Netherlands"
        )
        assert result["category"] == "legal_contract"

    def test_predict_hr_document(self, trained_classifier):
        result = trained_classifier.predict(
            "All employees are entitled to 25 days of annual leave per year"
        )
        assert result["category"] == "hr_policy"

    def test_predict_technical_document(self, trained_classifier):
        result = trained_classifier.predict(
            "The microservices architecture uses Kubernetes with auto-scaling"
        )
        assert result["category"] == "technical_spec"

    def test_predict_marketing_document(self, trained_classifier):
        result = trained_classifier.predict(
            "The campaign targets C-suite decision makers with social media strategy"
        )
        assert result["category"] == "marketing_brief"

    def test_predict_raises_if_not_trained(self):
        clf = DocumentClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict("some text")


class TestDocumentClassifierBatch:
    def test_predict_batch_returns_list(self, trained_classifier):
        texts = ["Revenue grew 15%", "This contract shall commence"]
        results = trained_classifier.predict_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_predict_batch_result_keys(self, trained_classifier):
        results = trained_classifier.predict_batch(["Revenue grew 15%"])
        assert "category" in results[0]
        assert "confidence" in results[0]
        assert "text" in results[0]

    def test_predict_batch_raises_if_not_trained(self):
        clf = DocumentClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict_batch(["some text"])


class TestTopFeatures:
    def test_get_top_features_returns_dict(self, trained_classifier):
        features = trained_classifier.get_top_features(n=5)
        assert isinstance(features, dict)
        assert len(features) == len(CATEGORIES)

    def test_top_features_have_expected_keys(self, trained_classifier):
        features = trained_classifier.get_top_features(n=3)
        for category, feat_list in features.items():
            assert len(feat_list) == 3
            for feat in feat_list:
                assert "feature" in feat
                assert "weight" in feat

    def test_top_features_raises_if_not_trained(self):
        clf = DocumentClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.get_top_features()
