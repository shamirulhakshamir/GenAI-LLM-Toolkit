"""
Document Classifier - LangGraph Multi-Agent Orchestration & Routing Pipeline.
Orchestrates enterprise document intelligence using a supervisor-agent topology,
state-driven task distribution, and conditional routing nodes.
"""

from typing import Dict, List, Literal, TypedDict
import json
from langgraph.graph import StateGraph, END


# 1. Define the Shared Agent State Schema
class AgentState(TypedDict):
    """Maintains conversation context, routing decisions, and execution metadata."""
    document_text: str
    inferred_intent: str
    assigned_analyst: str
    validation_passed: bool
    final_payload: Dict[str, any]
    execution_logs: List[str]


# 2. Simulated LLM-based Agent Nodes
class DocumentOrchestrator:
    """Supervisor agent that analyzes text intent and assigns specialized workers."""

    def analyze_intent(self, state: AgentState) -> Dict[str, any]:
        text = state["document_text"].lower()
        logs = state.get("execution_logs", [])
        logs.append("[Supervisor] Analyzing incoming document structure and semantic keys.")

        # Simulate dynamic structural classification patterns
        if "revenue" in text or "margin" in text or "ebitda" in text:
            intent = "financial_report"
            analyst = "FinanceAnalystAgent"
        elif "agreement" in text or "governed" in text or "breach" in text:
            intent = "legal_contract"
            analyst = "LegalComplianceAgent"
        elif "employee" in text or "leave" in text or "hr" in text:
            intent = "hr_policy"
            analyst = "HRPolicyAgent"
        elif "architecture" in text or "api" in text or "database" in text:
            intent = "technical_spec"
            analyst = "ArchitectureAgent"
        else:
            intent = "marketing_brief"
            analyst = "BrandStrategistAgent"

        logs.append(f"[Supervisor] Inferred category: {intent}. Routing to: {analyst}.")
        
        return {
            "inferred_intent": intent,
            "assigned_analyst": analyst,
            "execution_logs": logs
        }


class SpecialistWorkerNodes:
    """Contains localized worker routines matching specific business contexts."""

    def process_financial(self, state: AgentState) -> Dict[str, any]:
        logs = state["execution_logs"]
        logs.append("[FinanceAnalyst] Running dynamic extraction on financial terms.")
        return {
            "final_payload": {"processed_by": "FinanceAnalyst", "status": "COMPLETED", "confidence": 0.96},
            "execution_logs": logs
        }

    def process_legal(self, state: AgentState) -> Dict[str, any]:
        logs = state["execution_logs"]
        logs.append("[LegalCompliance] Parsing compliance variables and liability clauses.")
        return {
            "final_payload": {"processed_by": "LegalCompliance", "status": "COMPLETED", "confidence": 0.94},
            "execution_logs": logs
        }

    def process_general(self, state: AgentState) -> Dict[str, any]:
        logs = state["execution_logs"]
        logs.append(f"[{state['assigned_analyst']}] Extracting key entities and metadata.")
        return {
            "final_payload": {"processed_by": state["assigned_analyst"], "status": "COMPLETED", "confidence": 0.89},
            "execution_logs": logs
        }


# 3. Dynamic Conditional Routing Core Engine
def route_to_specialist(state: AgentState) -> Literal["finance", "legal", "general"]:
    """Evaluates graph state on the fly to select the next processing boundary."""
    intent = state["inferred_intent"]
    if intent == "financial_report":
        return "finance"
    elif intent == "legal_contract":
        return "legal"
    return "general"


class DocumentClassifier:
    """LangGraph multi-step task orchestration pipeline for text intelligence."""

    def __init__(self):
        self.orchestrator = DocumentOrchestrator()
        self.workers = SpecialistWorkerNodes()
        self.graph = self._build_execution_graph()

    def _build_execution_graph(self):
        """Builds a state-driven computational graph layout."""
        builder = StateGraph(AgentState)

        # Register execution nodes
        builder.add_node("SupervisorNode", self.orchestrator.analyze_intent)
        builder.add_node("FinanceWorkerNode", self.workers.process_financial)
        builder.add_node("LegalWorkerNode", self.workers.process_legal)
        builder.add_node("GeneralWorkerNode", self.workers.process_general)

        # Establish entry points and conditional branches
        builder.set_entry_point("SupervisorNode")
        builder.add_conditional_edges(
            "SupervisorNode",
            route_to_specialist,
            {
                "finance": "FinanceWorkerNode",
                "legal": "LegalWorkerNode",
                "general": "GeneralWorkerNode"
            }
        )

        # Wire endpoints back to completion states
        builder.add_edge("FinanceWorkerNode", END)
        builder.add_edge("LegalWorkerNode", END)
        builder.add_edge("GeneralWorkerNode", END)

        return builder.compile()

    def run_workflow(self, document_text: str) -> Dict[str, any]:
        """Triggers the execution graph with input variables and steps through the state."""
        initial_state: AgentState = {
            "document_text": document_text,
            "inferred_intent": "",
            "assigned_analyst": "",
            "validation_passed": True,
            "final_payload": {},
            "execution_logs": []
        }
        
        # Execute the active execution loop
        final_output = self.graph.invoke(initial_state)
        
        return {
            "category": final_output["inferred_intent"],
            "confidence": final_output["final_payload"]["confidence"],
            "logs": final_output["execution_logs"],
            "metadata": final_output["final_payload"]
        }


if __name__ == "__main__":
    # Validate orchestration pattern locally
    agent = DocumentClassifier()
    
    test_docs = [
        "The quarterly earnings report shows a 15% increase in net profit margin",
        "The API gateway handles authentication using OAuth 2.0 bearer tokens",
        "This agreement is binding upon execution by both parties hereto",
    ]

    for doc in test_docs:
        print(f"Input Text: '{doc}'")
        result = agent.run_workflow(doc)
        print(json.dumps(result, indent=2))
        print("-" * 60)
