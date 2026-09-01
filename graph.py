"""
graph.py

LangGraph StateGraph wiring all 5 SupportIQ agents together.

Flow:
    START → Intent Classifier → RAG Retrieval → Response Generator
           → Quality Evaluator → [conditional edge]
                                  ├── PASS or retries exhausted → HITL Reviewer → END
                                  └── BLOCK, retries remain → back to Response Generator
"""

from langgraph.graph import StateGraph, START, END
from pii_masking import mask_pii
from state import SupportIQState
from agents.intent_classifier import intent_classifier_node
from agents.rag_retrieval import rag_retrieval_node
from agents.response_generator import response_generator_node
from agents.quality_evaluator import quality_evaluator_node
from agents.hitl_reviewer import hitl_reviewer_node
from config import GROQ_API_KEY, MAX_RETRIES

def build_graph():
    """
    Constructs the SupportIQ LangGraph StateGraph with all 5 agents
    and the conditional retry/escalation routing logic.
    """
    graph = StateGraph(SupportIQState)

    # --- Register all 5 nodes ---
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("response_generator", response_generator_node)
    graph.add_node("quality_evaluator", quality_evaluator_node)
    graph.add_node("hitl_reviewer", hitl_reviewer_node)

    # --- Simple edges: fixed sequence, no decision needed ---
    graph.add_edge(START, "intent_classifier")
    graph.add_edge("intent_classifier", "rag_retrieval")
    graph.add_edge("rag_retrieval", "response_generator")
    graph.add_edge("response_generator", "quality_evaluator")

    # --- Conditional edge: routing decision after quality evaluation ---
    graph.add_conditional_edges(
        "quality_evaluator",
        route_after_quality_check,
        {
            "retry": "response_generator",
            "proceed": "hitl_reviewer",
        }
    )

    # --- HITL Reviewer always ends the pipeline ---
    graph.add_edge("hitl_reviewer", END)

    return graph.compile()


def route_after_quality_check(state: SupportIQState) -> str:
    """
    Decides where to go after Agent 4 (Quality Evaluator) runs.

    Returns:
        "proceed" — go to HITL Reviewer (quality passed, or retries exhausted)
        "retry"   — go back to Response Generator (quality failed, retries remain)
    """
    quality_passed = state.get("quality_passed", False)
    retry_count = state.get("retry_count", 0)

    if quality_passed:
        return "proceed"

    if retry_count >= MAX_RETRIES:
        return "proceed"  # exhausted retries — escalate to human anyway

    return "retry"

def run_pipeline(ticket: str, customer_id: str) -> dict:
    """
    Entry point for running the full SupportIQ pipeline.

    Args:
        ticket: raw customer ticket text
        customer_id: customer identifier for memory lookup

    Returns:
        Full final state dict after the pipeline completes —
        includes draft_response, quality_scores, escalated flag,
        retrieved_chunks, etc. for Streamlit to display.
    """
    # Mask PII before the ticket enters the pipeline
    masked_ticket = mask_pii(ticket)

    if masked_ticket != ticket:
        print("Pipeline → PII detected and masked in ticket")

    app = build_graph()

    initial_state = {
        "ticket": masked_ticket,
        "customer_id": customer_id,
        "retry_count": 0,
        "intent": None,
        "priority": None,
        "retrieved_chunks": None,
        "draft_response": None,
        "quality_scores": None,
        "quality_passed": None,
        "rejection_reason": None,
        "final_response": None,
        "hitl_decision": None,
        "escalated": None,
    }

    print(f"\n{'=' * 60}")
    print(f"Pipeline started for customer {customer_id}")
    print(f"{'=' * 60}\n")

    final_state = app.invoke(initial_state)

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete — escalated: {final_state.get('escalated')}")
    print(f"{'=' * 60}\n")

    return final_state

if __name__ == "__main__":
    from memory import init_db
    init_db()

    result = run_pipeline(
        ticket="What is my liability if I report fraud after 5 days?",
        customer_id="CUST-003",
    )

    print("FINAL RESULT SUMMARY:")
    print(f"Intent: {result.get('intent')}")
    print(f"Priority: {result.get('priority')}")
    print(f"Retry count: {result.get('retry_count')}")
    print(f"Quality passed: {result.get('quality_passed')}")
    print(f"Escalated: {result.get('escalated')}")
    print(f"\nDraft Response:\n{result.get('draft_response')}")