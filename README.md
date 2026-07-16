# GenAI-LLM-Toolkit

An enterprise-grade LLM orchestration and agentic systems toolkit featuring multi-step task orchestration, secure custom protocol servers, and schema-enforced output pipelines — built to demonstrate production-grade AI engineering design patterns.

## Project Structure

```text
genai-llm-toolkit/
├── src/
│   ├── document_classifier.py    - LangGraph multi-agent orchestration & routing
│   ├── insight_extractor.py      - Custom MCP server & tool data integration
│   └── report_automator.py       - LCEL execution chain & Pydantic structured output
├── tests/
│   ├── test_document_classifier.py
│   ├── test_insight_extractor.py
│   └── test_report_automator.py
├── requirements.txt
└── README.md
```

## Components

### 1. Agent Logic & Workflows (`src/document_classifier.py`)
Architects a multi-agent orchestration system using **LangGraph** to process complex internal business workflows. Implements:
- **State-driven task orchestration** and cyclical execution loops for complex document routing.
- **Conditional routing nodes** that evaluate text intent to dynamically delegate tasks across specialized LLM modules.
- **Human-in-the-Loop gates** designed to pause execution for manual verification on high-impact state transitions.

### 2. Context Layer & Custom Protocols (`src/insight_extractor.py`)
Implements native **Model Context Protocol (MCP)** servers to bridge LLM capability with local data layers and infrastructure. Implements:
- **Exposed Enterprise Tools:** Wraps internal file paths and system endpoints into unified schemas consumable by agents.
- **Scoped Token Authentication:** Enforces strict security layers, mapping the LLM session context directly to explicit database access levels to prevent prompt injections and data leaks.

### 3. Execution & Structured Data Output (`src/report_automator.py`)
Standardizes model communication and data extraction pipelines to ensure reliable backend integrations. Implements:
- **LangChain Expression Language (LCEL):** Streamlined runtime chains using parallel execution blocks and robust model failover configurations.
- **Pydantic Validation Engines:** Enforces rigid schema parsing to guarantee LLM payloads compile into predictable, minified JSON before execution.

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Running Individual Components

python
from src.document_classifier import DocumentClassifier
from src.insight_extractor import InsightExtractor
from src.report_automator import ReportAutomator

# 1. Orchestrate agent workflows using state-driven routing
agent = DocumentClassifier()
result = agent.run_workflow("Process internal document validation for employee reference ID: HR-9081.")

# 2. Expose secure data context via MCP server bindings
context_layer = InsightExtractor()
mcp_tools = context_layer.initialize_mcp_server(scoped_token="env_session_token")

# 3. Parse and enforce strict output constraints on payloads
validator = ReportAutomator()
validated_json = validator.generate_structured_payload(raw_agent_output=result)
print(validated_json)


## Tech Stack

- *Python 3.10+*
- *LangGraph* - State management, multi-agent systems, conditional routing
- *LangChain & LCEL* - Tool-calling, Pydantic data extraction, runtime failover
- *Model Context Protocol (MCP)* - Secure context layers, client-server tool schemas
- *Pytest* - Automated regression testing suites

## Author

Shamirul Hak Surbudeen
