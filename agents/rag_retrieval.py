"""
agents/rag_retrieval.py

Agent 2 in the SupportIQ pipeline.
Runs hybrid search (dense + sparse + RRF) over NovaPay policy documents
and writes the top-3 retrieved chunks to shared state.

No LLM call — hybrid_search() is deterministic math.
Wrapping as a LangGraph node gives:
    - Consistent pipeline structure
    - LangSmith tracing at this step
    - Future flexibility to add query rewriting before search
"""

from state import SupportIQState
from rag.hybrid import hybrid_search

# Sentinel chunk returned when no relevant policy found.
# Keeps downstream agents on a single code path — no empty list special casing.
NO_RESULTS_SENTINEL = {
    "content": "No relevant policy documents found for this query. "
               "Please escalate to a human agent for manual policy lookup.",
    "source": "system",
    "score": 0.0,
    "retriever": "none",
}

# Intent-specific search term enrichment.
# Appended to ticket text to improve retrieval for short/vague tickets.
# Terms chosen to match vocabulary in NovaPay policy documents.
INTENT_CONTEXT = {
    "refund": "refund policy transaction reversal reimbursement",
    "technical": "UPI NEFT IMPS transaction failure resolution error",
    "complaint": "grievance complaint resolution dissatisfied service",
    "inquiry": "account policy information eligibility terms conditions",
    "escalation": "escalation grievance complaint senior review unresolved",
}


def rag_retrieval_node(state: SupportIQState) -> dict:
    """
    Agent 2: Retrieves relevant policy chunks using hybrid search.

    Reads from state:  ticket, intent (for optional query enrichment)
    Writes to state:   retrieved_chunks

    No LLM call — hybrid_search() is deterministic math.
    Returns partial state update — only retrieved_chunks.
    """
    ticket = state["ticket"]
    intent = state.get("intent", "")

    # Build enriched query — append intent context if available
    if intent and intent in INTENT_CONTEXT:
        enriched_query = f"{ticket} {INTENT_CONTEXT[intent]}"
        print(f"RAG Retrieval → enriched query with intent: {intent}")
    else:
        enriched_query = ticket
        print("RAG Retrieval → using raw ticket as query")

    try:
        chunks = hybrid_search(enriched_query)

        # No chunks found — return sentinel instead of empty list
        if not chunks:
            print("RAG Retrieval → no chunks found, using sentinel")
            return {"retrieved_chunks": [NO_RESULTS_SENTINEL]}

        print(f"RAG Retrieval → retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}: {chunk['source']} "
                  f"(rrf_score: {chunk['rrf_score']}, "
                  f"retrievers: {chunk['retrievers']})")

        return {"retrieved_chunks": chunks}

    except Exception as e:
        # hybrid_search failed — Chroma or BM25 issue
        print(f"RAG Retrieval → failed ({e}), using sentinel")
        return {"retrieved_chunks": [NO_RESULTS_SENTINEL]}


if __name__ == "__main__":
    # Standalone test — same ticket as hybrid.py test for direct comparison
    test_cases = [
        {
            "ticket": "What is my liability if I report fraud after 5 days?",
            "intent": "technical",
            "expected_source": "08_fraud_claim_escalation.txt"
        },
        {
            "ticket": "I want to close my savings account",
            "intent": "inquiry",
            "expected_source": "01_savings_account_terms.txt"
        },
    ]

    for test in test_cases:
        print(f"\nTicket: {test['ticket']}")
        print(f"Intent: {test['intent']}")
        print(f"Expected source: {test['expected_source']}")
        print("-" * 50)

        from state import SupportIQState
        state = SupportIQState(
            ticket=test["ticket"],
            customer_id="TEST-001",
            retry_count=0,
            intent=test["intent"],
            priority="high",
            retrieved_chunks=None,
            draft_response=None,
            quality_scores=None,
            quality_passed=None,
            rejection_reason=None,
            final_response=None,
            hitl_decision=None,
        )

        result = rag_retrieval_node(state)
        chunks = result["retrieved_chunks"]
        print(f"Got {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  {i}. {chunk['source']}")
            print(f"     {chunk['content'][:100]}...")
        print()