"""
Document Classifier - TF-IDF based document classification pipeline.
Classifies business documents into categories using TF-IDF vectorization
and logistic regression.
"""

import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


# Synthetic training data for business document classification
SYNTHETIC_DOCUMENTS = {
    "financial_report": [
        "Q3 revenue increased by 12% year-over-year driven by strong performance in APAC markets",
        "Operating margin improved to 18.5% reflecting cost optimization initiatives and volume growth",
        "Total assets grew to $4.2 billion with cash reserves of $890 million at quarter end",
        "EBITDA reached $156 million representing a 15% improvement over the prior year period",
        "Net income attributable to shareholders was $92 million or $1.45 per diluted share",
        "Free cash flow generation of $210 million supported by working capital improvements",
        "Revenue guidance for FY2025 raised to $3.8-4.0 billion based on strong pipeline visibility",
        "Gross profit margin expanded 200 basis points to 42% driven by favorable product mix",
        "The company reported quarterly earnings of $2.15 per share exceeding analyst estimates",
        "Capital expenditure totaled $145 million primarily for capacity expansion in emerging markets",
        "Annual dividend increased by 8% to $0.92 per share reflecting confidence in cash generation",
        "Accounts receivable turnover improved to 45 days from 52 days in the prior year",
    ],
    "legal_contract": [
        "This agreement shall be governed by and construed in accordance with the laws of the Netherlands",
        "The parties agree to maintain confidentiality of all proprietary information for a period of five years",
        "In the event of breach the non-breaching party shall be entitled to seek injunctive relief",
        "The indemnifying party shall hold harmless the indemnified party from any third-party claims",
        "This contract shall commence on the effective date and continue for an initial term of three years",
        "Either party may terminate this agreement upon ninety days written notice to the other party",
        "The contractor agrees to comply with all applicable data protection regulations including GDPR",
        "Force majeure events shall excuse performance obligations for the duration of the event",
        "Disputes arising under this agreement shall be resolved through binding arbitration in Amsterdam",
        "The licensee is granted a non-exclusive worldwide license to use the software for internal purposes",
        "Representations and warranties made herein shall survive the closing of the transaction",
        "The purchase price shall be adjusted based on the final working capital calculation at closing",
    ],
    "hr_policy": [
        "All employees are entitled to 25 days of annual leave plus public holidays per calendar year",
        "The company offers a hybrid working arrangement with minimum three days in-office per week",
        "Performance reviews are conducted semi-annually with mid-year check-ins and year-end evaluations",
        "New employees must complete mandatory onboarding training within the first 30 days of employment",
        "The company provides comprehensive health insurance coverage for employees and their dependents",
        "Parental leave policy provides 16 weeks of fully paid leave for primary caregivers",
        "Professional development budget of EUR 2000 per employee per year for courses and certifications",
        "The grievance procedure requires employees to first raise concerns with their direct manager",
        "Remote work requests must be approved by the department head and HR business partner",
        "The company maintains a zero-tolerance policy for workplace harassment and discrimination",
        "Employee stock purchase plan allows staff to acquire shares at a 15% discount to market price",
        "Retirement contributions are matched at 50% up to 6% of base salary for all eligible employees",
    ],
    "technical_spec": [
        "The system architecture uses microservices deployed on Kubernetes with auto-scaling enabled",
        "API endpoints must respond within 200ms at the 95th percentile under normal load conditions",
        "Data pipeline processes 500GB daily using Apache Spark with checkpointing for fault tolerance",
        "The database schema includes normalized tables with foreign key constraints and indexed lookups",
        "Authentication is handled via OAuth 2.0 with JWT tokens having a 15-minute expiration window",
        "The CI/CD pipeline runs unit tests integration tests and security scans before deployment",
        "Service mesh implemented with Istio for traffic management and mutual TLS between services",
        "Event-driven architecture uses Kafka topics with guaranteed at-least-once delivery semantics",
        "The monitoring stack includes Prometheus for metrics Grafana for dashboards and PagerDuty alerts",
        "Container images are scanned for CVEs and must pass security baseline before registry push",
        "Load balancing uses round-robin with health checks every 10 seconds and 3 failure threshold",
        "Database replication configured with synchronous primary and two asynchronous read replicas",
    ],
    "marketing_brief": [
        "The campaign targets C-suite decision makers in financial services across Western Europe",
        "Brand positioning emphasizes innovation trust and digital transformation capabilities",
        "Social media strategy focuses on LinkedIn thought leadership and targeted sponsored content",
        "Expected reach of 2.5 million impressions with a target conversion rate of 3.2%",
        "Content pillars include sustainability digital innovation and client success stories",
        "The product launch event will feature keynote speakers from three Fortune 500 client companies",
        "Email nurture sequence consists of six touchpoints over a 45-day engagement window",
        "Competitive analysis shows our differentiation in AI-powered analytics and consulting depth",
        "Budget allocation splits 40% digital advertising 30% events and 30% content production",
        "Customer persona research identified five key segments based on company size and industry",
        "A/B testing of landing pages showed 28% higher conversion with video-first design approach",
        "Influencer partnership program engages 15 industry thought leaders for quarterly content",
    ],
}

CATEGORIES = list(SYNTHETIC_DOCUMENTS.keys())


class DocumentClassifier:
    """TF-IDF + Logistic Regression document classifier for business documents."""

    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self.is_trained = False
        self.categories = CATEGORIES

    def _prepare_data(self):
        """Prepare training data from synthetic documents."""
        texts = []
        labels = []
        for category, docs in SYNTHETIC_DOCUMENTS.items():
            for doc in docs:
                texts.append(doc)
                labels.append(category)
        return texts, labels

    def train(self, test_size=0.2):
        """Train the classifier on synthetic data. Returns accuracy on test split."""
        texts, labels = self._prepare_data()
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)
        self.is_trained = True

        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        return {
            "accuracy": accuracy,
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    def predict(self, text):
        """Classify a single document. Returns category and confidence scores."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = dict(zip(self.model.classes_, probabilities.tolist()))
        return {
            "category": prediction,
            "confidence": max(probabilities),
            "all_scores": confidence,
        }

    def predict_batch(self, texts):
        """Classify multiple documents at once."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        results = []
        for i, text in enumerate(texts):
            confidence = dict(zip(self.model.classes_, probabilities[i].tolist()))
            results.append({
                "text": text[:80] + "..." if len(text) > 80 else text,
                "category": predictions[i],
                "confidence": max(probabilities[i]),
                "all_scores": confidence,
            })
        return results

    def get_top_features(self, n=10):
        """Get top TF-IDF features per category."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        feature_names = self.vectorizer.get_feature_names_out()
        top_features = {}
        for i, category in enumerate(self.model.classes_):
            coef = self.model.coef_[i]
            top_indices = np.argsort(coef)[-n:][::-1]
            top_features[category] = [
                {"feature": feature_names[idx], "weight": float(coef[idx])}
                for idx in top_indices
            ]
        return top_features


if __name__ == "__main__":
    clf = DocumentClassifier()
    results = clf.train()
    print(f"Training accuracy: {results['accuracy']:.2%}")
    print()

    test_docs = [
        "The quarterly earnings report shows a 15% increase in net profit margin",
        "All employees must complete the annual compliance training by December 31",
        "The API gateway handles authentication using OAuth 2.0 bearer tokens",
        "This agreement is binding upon execution by both parties hereto",
        "Our social media campaign generated 1.2 million impressions last quarter",
    ]

    for doc in test_docs:
        result = clf.predict(doc)
        print(f"Text: {doc[:70]}...")
        print(f"  Category: {result['category']} (confidence: {result['confidence']:.2%})")
        print()
