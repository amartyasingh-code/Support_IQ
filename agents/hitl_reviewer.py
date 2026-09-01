"""
agents/hitl_reviewer.py

Agent 5 in the SupportIQ pipeline — the final step before response delivery.

Unlike Agents 1-4, this node does NOT make an automated decision.
Its job is narrow: detect whether this ticket reached HITL via normal
PASS or via forced escalation (retries exhausted), and write that single
flag to state. The actual human review interaction happens in app.py
(Streamlit) — this node just prepares the routing signal.

Streamlit reads existing state fields directly (ticket, draft_response,
quality_scores, retrieved_chunks) rather than a bundled summary — avoids
duplicating data that could go stale across retries.
"""

from state import SupportIQState

from config import MAX_RETRIES


def hitl_reviewer_node(state: SupportIQState) -> dict:
    """
    Agent 5: Prepares the escalation flag for human review.

    Reads from state:  quality_passed, retry_count
    Writes to state:   escalated (bool — True if forced due to
                       exhausted retries, False if normal PASS)

    Does NOT set final_response or hitl_decision — those are set
    by Streamlit when the human actually clicks Approve/Edit/Reject.
    """
    quality_passed = state.get("quality_passed", False)
    retry_count = state.get("retry_count", 0)

    # Escalated = reached HITL without ever passing quality checks
    # (i.e., retries were exhausted, not because quality genuinely passed)
    escalated = not quality_passed and retry_count >= MAX_RETRIES

    if escalated:
        print(f"HITL Reviewer → ESCALATED (retries exhausted after "
              f"{retry_count} attempts) — flagging for priority human review")
    else:
        print("HITL Reviewer → normal review (quality checks passed)")

    return {"escalated": escalated}

if __name__ == "__main__":
    from state import SupportIQState

    # Test case 1: normal PASS
    state_normal = SupportIQState(
        ticket="test",
        customer_id="CUST-001",
        retry_count=0,
        intent="inquiry",
        priority="low",
        retrieved_chunks=[],
        draft_response="test response",
        quality_scores={"faithfulness": 0.9, "relevance": 0.85},
        quality_passed=True,
        rejection_reason=None,
        final_response=None,
        hitl_decision=None,
        escalated=None,
    )

    print("Test 1: Normal PASS")
    result1 = hitl_reviewer_node(state_normal)
    print(f"Result: {result1}\n")

    # Test case 2: escalated after 2 failed retries
    state_escalated = SupportIQState(
        ticket="test",
        customer_id="CUST-001",
        retry_count=2,
        intent="technical",
        priority="critical",
        retrieved_chunks=[],
        draft_response="test response that keeps failing",
        quality_scores={"faithfulness": 0.3, "relevance": 0.4},
        quality_passed=False,
        rejection_reason="Faithfulness too low",
        final_response=None,
        hitl_decision=None,
        escalated=None,
    )

    print("Test 2: Escalated after exhausted retries")
    result2 = hitl_reviewer_node(state_escalated)
    print(f"Result: {result2}")