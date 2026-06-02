# GenAI-LLM-Toolkit

.An NLP and document intelligence toolkit featuring automated document classification, insight extraction, and report generation 
— built to demonstrate production-grade NLP pipeline design.

## Project Structure

```
operational-intelligence-nlp-toolkit/
  src/
    document_classifier.py   - TF-IDF document classification pipeline
    insight_extractor.py      - Automated insight extraction from text
    report_automator.py       - Automated consulting report generation
  tests/
    test_document_classifier.py
    test_insight_extractor.py
    test_report_automator.py
  requirements.txt
  README.md
```

## Components

### 1. Document Classifier (`src/document_classifier.py`)
Classifies business documents into categories (financial reports, legal contracts, HR policies, technical specs, marketing briefs) using TF-IDF vectorization and logistic regression. Includes synthetic training data generation and a full train/predict pipeline.

### 2. Insight Extractor (`src/insight_extractor.py`)
Extracts structured insights from unstructured business text: key metrics (percentages, currency, counts), named entities, sentiment indicators, and action items. Uses regex-based NLP patterns for reliable extraction without external API dependencies.

### 3. Report Automator (`src/report_automator.py`)
Generates structured consulting reports from extracted insights and classification results. Produces executive summaries, key findings, and recommendations in a standardized format suitable for client delivery.

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Running Individual Components

```python
from src.document_classifier import DocumentClassifier
from src.insight_extractor import InsightExtractor
from src.report_automator import ReportAutomator

# Classify a document
clf = DocumentClassifier()
clf.train()
result = clf.predict("Q3 revenue increased by 12% driven by strong APAC growth")

# Extract insights
extractor = InsightExtractor()
insights = extractor.extract("Revenue grew 15% to $2.3M with 500 new customers")

# Generate report
automator = ReportAutomator()
report = automator.generate(classification=result, insights=insights)
print(report)
```

## Tech Stack

- **Python 3.10+**
- **Scikit-learn** - TF-IDF vectorization, logistic regression
- **Pandas/NumPy** - Data manipulation
- **NLTK** - Text preprocessing
- **Pytest** - Testing framework

## Author

Shamirul Hak Surbudeen
