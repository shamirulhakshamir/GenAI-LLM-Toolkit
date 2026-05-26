"""
Report Automator - Automated consulting report generation.
Generates structured consulting reports from classification results
and extracted insights in a standardized format for client delivery.
"""

from datetime import datetime


class ReportAutomator:
    """Generates structured consulting reports from document analysis results."""

    CATEGORY_LABELS = {
        "financial_report": "Financial Report",
        "legal_contract": "Legal Contract",
        "hr_policy": "HR Policy Document",
        "technical_spec": "Technical Specification",
        "marketing_brief": "Marketing Brief",
    }

    CATEGORY_RECOMMENDATIONS = {
        "financial_report": [
            "Conduct detailed variance analysis on key financial metrics",
            "Benchmark performance against industry peers and historical trends",
            "Assess risk factors that could impact future financial projections",
        ],
        "legal_contract": [
            "Review compliance with applicable regulatory requirements",
            "Identify potential liability exposure and mitigation strategies",
            "Ensure alignment with organizational risk appetite and governance framework",
        ],
        "hr_policy": [
            "Evaluate policy effectiveness through employee satisfaction surveys",
            "Benchmark compensation and benefits against market standards",
            "Assess alignment with current labor regulations and best practices",
        ],
        "technical_spec": [
            "Conduct architectural review for scalability and security posture",
            "Evaluate alignment with enterprise technology standards",
            "Assess technical debt and recommend modernization priorities",
        ],
        "marketing_brief": [
            "Analyze campaign ROI and optimize channel allocation",
            "Conduct competitive positioning analysis and brand perception study",
            "Develop data-driven targeting strategy based on customer segmentation",
        ],
    }

    def __init__(self, analyst_name="Shamirul Hak Surbudeen", firm_name="GenAI Toolkit"):
        self.analyst_name = analyst_name
        self.firm_name = firm_name

    def generate(self, classification, insights, document_title="Analyzed Document"):
        """
        Generate a full consulting report.

        Args:
            classification: dict with 'category' and 'confidence' from DocumentClassifier
            insights: dict from InsightExtractor.extract()
            document_title: optional title for the document

        Returns:
            str: formatted consulting report
        """
        sections = [
            self._header(document_title),
            self._executive_summary(classification, insights),
            self._classification_section(classification),
            self._key_findings(insights),
            self._metrics_section(insights),
            self._sentiment_section(insights),
            self._action_items_section(insights),
            self._recommendations(classification),
            self._footer(),
        ]
        return "\n".join(sections)

    def generate_summary(self, classification, insights):
        """Generate a brief executive summary only (for quick reviews)."""
        category = classification.get("category", "unknown")
        confidence = classification.get("confidence", 0)
        label = self.CATEGORY_LABELS.get(category, category)
        sentiment = insights.get("sentiment", {})
        metrics = insights.get("metrics", {})
        stats = insights.get("summary_stats", {})

        num_metrics = (
            len(metrics.get("percentages", []))
            + len(metrics.get("currencies", []))
            + len(metrics.get("counts", []))
        )

        return (
            f"Document classified as '{label}' with {confidence:.0%} confidence. "
            f"Sentiment: {sentiment.get('label', 'neutral')} (score: {sentiment.get('score', 0):.2f}). "
            f"Extracted {num_metrics} quantitative metrics from "
            f"{stats.get('word_count', 0)} words across {stats.get('sentence_count', 0)} sentences."
        )

    def _header(self, document_title):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "=" * 72,
            f"  {self.firm_name} | CONSULTING ANALYSIS REPORT",
            "=" * 72,
            "",
            f"  Document:  {document_title}",
            f"  Analyst:   {self.analyst_name}",
            f"  Date:      {now}",
            "",
            "-" * 72,
        ]
        return "\n".join(lines)

    def _executive_summary(self, classification, insights):
        summary_text = self.generate_summary(classification, insights)
        action_items = insights.get("action_items", [])
        lines = [
            "",
            "  EXECUTIVE SUMMARY",
            "  " + "-" * 40,
            "",
            f"  {summary_text}",
            "",
        ]
        if action_items:
            lines.append(f"  {len(action_items)} action item(s) identified requiring attention.")
        lines.append("")
        return "\n".join(lines)

    def _classification_section(self, classification):
        category = classification.get("category", "unknown")
        label = self.CATEGORY_LABELS.get(category, category)
        confidence = classification.get("confidence", 0)
        all_scores = classification.get("all_scores", {})

        lines = [
            "  DOCUMENT CLASSIFICATION",
            "  " + "-" * 40,
            "",
            f"  Primary Category:  {label}",
            f"  Confidence:        {confidence:.1%}",
            "",
        ]

        if all_scores:
            lines.append("  Category Probabilities:")
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            for cat, score in sorted_scores:
                cat_label = self.CATEGORY_LABELS.get(cat, cat)
                bar_len = int(score * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                lines.append(f"    {cat_label:30s} [{bar}] {score:.1%}")
            lines.append("")

        return "\n".join(lines)

    def _key_findings(self, insights):
        metrics = insights.get("metrics", {})
        sentiment = insights.get("sentiment", {})
        key_terms = insights.get("word_frequency", [])

        findings = []

        pcts = metrics.get("percentages", [])
        if pcts:
            top_pct = max(pcts, key=lambda x: x["value"])
            findings.append(
                f"Highest percentage metric identified: {top_pct['value']}%"
            )

        curs = metrics.get("currencies", [])
        if curs:
            top_cur = max(curs, key=lambda x: x["value"])
            findings.append(
                f"Largest monetary value referenced: {top_cur['raw']}"
            )

        if sentiment.get("label") != "neutral":
            findings.append(
                f"Document exhibits {sentiment['label']} sentiment "
                f"(score: {sentiment['score']:.2f})"
            )

        if key_terms:
            top_words = [term for term, _ in key_terms[:5]]
            findings.append(
                f"Key themes: {', '.join(top_words)}"
            )

        lines = [
            "  KEY FINDINGS",
            "  " + "-" * 40,
            "",
        ]
        if findings:
            for i, finding in enumerate(findings, 1):
                lines.append(f"  {i}. {finding}")
        else:
            lines.append("  No significant findings extracted.")
        lines.append("")
        return "\n".join(lines)

    def _metrics_section(self, insights):
        metrics = insights.get("metrics", {})
        lines = [
            "  QUANTITATIVE METRICS",
            "  " + "-" * 40,
            "",
        ]

        pcts = metrics.get("percentages", [])
        if pcts:
            lines.append("  Percentages:")
            for pct in pcts:
                lines.append(f"    - {pct['value']}%  |  {pct['context']}")
            lines.append("")

        curs = metrics.get("currencies", [])
        if curs:
            lines.append("  Monetary Values:")
            for cur in curs:
                lines.append(f"    - {cur['raw']} ({cur['value']:,.0f})  |  {cur['context']}")
            lines.append("")

        cnts = metrics.get("counts", [])
        if cnts:
            lines.append("  Counts:")
            for cnt in cnts:
                lines.append(f"    - {cnt['value']:,.0f} {cnt['entity']}")
            lines.append("")

        if not pcts and not curs and not cnts:
            lines.append("  No quantitative metrics found.")
            lines.append("")

        return "\n".join(lines)

    def _sentiment_section(self, insights):
        sentiment = insights.get("sentiment", {})
        lines = [
            "  SENTIMENT ANALYSIS",
            "  " + "-" * 40,
            "",
            f"  Overall Sentiment: {sentiment.get('label', 'N/A').upper()}",
            f"  Sentiment Score:   {sentiment.get('score', 0):.3f}  (-1 = negative, +1 = positive)",
            "",
        ]

        pos = sentiment.get("positive_indicators", [])
        if pos:
            lines.append(f"  Positive indicators: {', '.join(pos)}")

        neg = sentiment.get("negative_indicators", [])
        if neg:
            lines.append(f"  Negative indicators: {', '.join(neg)}")

        lines.append("")
        return "\n".join(lines)

    def _action_items_section(self, insights):
        actions = insights.get("action_items", [])
        lines = [
            "  ACTION ITEMS",
            "  " + "-" * 40,
            "",
        ]
        if actions:
            for i, action in enumerate(actions, 1):
                lines.append(f"  [{i}] {action}")
        else:
            lines.append("  No action items identified.")
        lines.append("")
        return "\n".join(lines)

    def _recommendations(self, classification):
        category = classification.get("category", "unknown")
        recs = self.CATEGORY_RECOMMENDATIONS.get(category, [])
        lines = [
            "  RECOMMENDATIONS",
            "  " + "-" * 40,
            "",
        ]
        if recs:
            for i, rec in enumerate(recs, 1):
                lines.append(f"  {i}. {rec}")
        else:
            lines.append("  No category-specific recommendations available.")
        lines.append("")
        return "\n".join(lines)

    def _footer(self):
        lines = [
            "-" * 72,
            f"  Report generated by {self.firm_name} GenAI Consulting Toolkit",
            f"  Analyst: {self.analyst_name}",
            "  CONFIDENTIAL - For internal use only",
            "=" * 72,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo with sample data
    sample_classification = {
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

    sample_insights = {
        "metrics": {
            "percentages": [
                {"value": 15.0, "context": "Revenue grew 15% year-over-year"},
                {"value": 22.5, "context": "Operating margin improved to 22.5%"},
            ],
            "currencies": [
                {"value": 2_300_000, "raw": "$2.3 million", "context": "revenue to $2.3 million"},
                {"value": 890_000, "raw": "$890 thousand", "context": "Net profit reached $890 thousand"},
            ],
            "counts": [
                {"value": 500, "entity": "customers"},
                {"value": 12000, "entity": "users"},
            ],
        },
        "sentiment": {
            "score": 0.6,
            "label": "positive",
            "positive_indicators": ["grew", "improved", "record", "strong"],
            "negative_indicators": ["risk"],
        },
        "action_items": [
            "develop a comprehensive expansion strategy for European markets",
            "implement automated reporting systems by Q1 2026",
            "review pricing structures to maintain competitive positioning",
        ],
        "key_dates": ["Q3 2025", "Q1 2026", "FY2026"],
        "word_frequency": [
            ("revenue", 3), ("market", 3), ("performance", 2),
            ("growth", 2), ("profit", 2),
        ],
        "summary_stats": {
            "word_count": 120,
            "sentence_count": 8,
            "avg_sentence_length": 15.0,
            "character_count": 650,
        },
    }

    automator = ReportAutomator()
    report = automator.generate(
        classification=sample_classification,
        insights=sample_insights,
        document_title="Q3 2025 Financial Performance Summary",
    )
    print(report)
