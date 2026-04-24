"""
Insight Extractor - Automated insight extraction from unstructured business text.
Uses regex-based NLP patterns to extract key metrics, entities, sentiment indicators,
and action items from business documents.
"""

import re
from collections import Counter


class InsightExtractor:
    """Extracts structured insights from unstructured business text."""

    # Patterns for metric extraction
    PERCENTAGE_PATTERN = re.compile(
        r'(\d+(?:\.\d+)?)\s*%', re.IGNORECASE
    )
    CURRENCY_PATTERN = re.compile(
        r'[\$\u00a3\u20ac][\s]?(\d+(?:[,.\d]*\d)?)\s*(million|billion|thousand|[MBKmk])?',
        re.IGNORECASE,
    )
    COUNT_PATTERN = re.compile(
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s+(customers?|users?|employees?|clients?|transactions?|units?|orders?|projects?)',
        re.IGNORECASE,
    )
    DATE_PATTERN = re.compile(
        r'(?:Q[1-4]\s*\d{4}|FY\s*\d{4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b|\b\d{4}\b)',
        re.IGNORECASE,
    )

    # Sentiment indicators
    POSITIVE_WORDS = {
        "increased", "improved", "grew", "growth", "gained", "exceeded",
        "strong", "positive", "optimistic", "expanded", "accelerated",
        "outperformed", "surpassed", "record", "highest", "best",
        "profitable", "successful", "efficient", "innovative",
    }
    NEGATIVE_WORDS = {
        "decreased", "declined", "fell", "dropped", "lost", "missed",
        "weak", "negative", "challenging", "contracted", "decelerated",
        "underperformed", "below", "lowest", "worst", "risk",
        "deficit", "failure", "inefficient", "concern",
    }

    # Action item patterns
    ACTION_PATTERNS = [
        re.compile(r'(?:must|should|need to|required to|will)\s+(.+?)(?:\.|$)', re.IGNORECASE),
        re.compile(r'(?:action|todo|next step|recommendation)[:\s]+(.+?)(?:\.|$)', re.IGNORECASE),
        re.compile(r'(?:ensure|implement|develop|create|establish|review)\s+(.+?)(?:\.|$)', re.IGNORECASE),
    ]

    def extract(self, text):
        """Extract all insights from text. Returns a structured dict."""
        return {
            "metrics": self.extract_metrics(text),
            "sentiment": self.analyze_sentiment(text),
            "action_items": self.extract_action_items(text),
            "key_dates": self.extract_dates(text),
            "word_frequency": self.get_key_terms(text),
            "summary_stats": self._compute_summary_stats(text),
        }

    def extract_metrics(self, text):
        """Extract numerical metrics: percentages, currencies, counts."""
        metrics = {
            "percentages": [],
            "currencies": [],
            "counts": [],
        }

        # Percentages
        for match in self.PERCENTAGE_PATTERN.finditer(text):
            value = float(match.group(1))
            # Get surrounding context (20 chars each side)
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()
            metrics["percentages"].append({
                "value": value,
                "context": context,
            })

        # Currencies
        for match in self.CURRENCY_PATTERN.finditer(text):
            raw_value = match.group(1).replace(",", "")
            value = float(raw_value)
            multiplier_text = match.group(2)
            if multiplier_text:
                ml = multiplier_text.lower()
                if ml in ("billion", "b"):
                    value *= 1_000_000_000
                elif ml in ("million", "m"):
                    value *= 1_000_000
                elif ml in ("thousand", "k"):
                    value *= 1_000
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()
            metrics["currencies"].append({
                "value": value,
                "raw": match.group(0),
                "context": context,
            })

        # Counts
        for match in self.COUNT_PATTERN.finditer(text):
            raw_value = match.group(1).replace(",", "")
            value = float(raw_value)
            entity = match.group(2).lower()
            metrics["counts"].append({
                "value": value,
                "entity": entity,
            })

        return metrics

    def analyze_sentiment(self, text):
        """Analyze overall sentiment based on keyword presence."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        positive_found = words & self.POSITIVE_WORDS
        negative_found = words & self.NEGATIVE_WORDS

        pos_count = len(positive_found)
        neg_count = len(negative_found)
        total = pos_count + neg_count

        if total == 0:
            score = 0.0
            label = "neutral"
        else:
            score = (pos_count - neg_count) / total
            if score > 0.2:
                label = "positive"
            elif score < -0.2:
                label = "negative"
            else:
                label = "neutral"

        return {
            "score": round(score, 3),
            "label": label,
            "positive_indicators": sorted(positive_found),
            "negative_indicators": sorted(negative_found),
        }

    def extract_action_items(self, text):
        """Extract action items and recommendations from text."""
        actions = []
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            for pattern in self.ACTION_PATTERNS:
                match = pattern.search(sentence)
                if match:
                    action_text = match.group(1).strip()
                    if len(action_text) > 10:  # Filter trivial matches
                        actions.append(action_text)
                    break  # One match per sentence
        # Deduplicate while preserving order
        seen = set()
        unique_actions = []
        for action in actions:
            normalized = action.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_actions.append(action)
        return unique_actions

    def extract_dates(self, text):
        """Extract date references from text."""
        dates = []
        for match in self.DATE_PATTERN.finditer(text):
            dates.append(match.group(0))
        return sorted(set(dates))

    def get_key_terms(self, text, top_n=10):
        """Get most frequent meaningful terms (excluding stopwords)."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "and",
            "but", "or", "nor", "not", "so", "yet", "both", "either",
            "neither", "each", "every", "all", "any", "few", "more",
            "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "because", "about", "up",
            "out", "if", "then", "that", "this", "it", "its", "per",
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stopwords]
        return Counter(filtered).most_common(top_n)

    def _compute_summary_stats(self, text):
        """Compute basic document statistics."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.split()
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": round(len(words) / max(len(sentences), 1), 1),
            "character_count": len(text),
        }


if __name__ == "__main__":
    sample_text = """
    Q3 2025 Financial Performance Summary

    Revenue grew 15% year-over-year to $2.3 million, driven by strong performance
    in the APAC region. Operating margin improved to 22.5% reflecting successful
    cost optimization initiatives. The company added 500 new customers during the
    quarter, bringing total active users to 12,000.

    Net profit reached $890 thousand, representing a record quarter. Free cash flow
    of $1.2 million exceeded expectations by 18%.

    Management must develop a comprehensive expansion strategy for European markets.
    The team should implement automated reporting systems by Q1 2026. We need to
    review pricing structures to maintain competitive positioning.

    Risk factors include potential regulatory changes and increased competition
    in core markets. However, the overall outlook remains positive with strong
    pipeline visibility into FY2026.
    """

    extractor = InsightExtractor()
    insights = extractor.extract(sample_text)

    print("=== Extracted Insights ===\n")

    print("METRICS:")
    for pct in insights["metrics"]["percentages"]:
        print(f"  Percentage: {pct['value']}% - {pct['context']}")
    for cur in insights["metrics"]["currencies"]:
        print(f"  Currency: {cur['raw']} (={cur['value']:,.0f}) - {cur['context']}")
    for cnt in insights["metrics"]["counts"]:
        print(f"  Count: {cnt['value']:,.0f} {cnt['entity']}")

    print(f"\nSENTIMENT: {insights['sentiment']['label']} (score: {insights['sentiment']['score']})")
    print(f"  Positive: {insights['sentiment']['positive_indicators']}")
    print(f"  Negative: {insights['sentiment']['negative_indicators']}")

    print(f"\nACTION ITEMS:")
    for action in insights["action_items"]:
        print(f"  - {action}")

    print(f"\nKEY DATES: {insights['key_dates']}")
    print(f"\nSUMMARY STATS: {insights['summary_stats']}")
