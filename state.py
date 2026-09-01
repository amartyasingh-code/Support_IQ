"""
state.py

Shared LangGraph state schema for the SupportIQ pipeline.
Every agent reads from and writes to this typed dictionary.

Design decisions:
- Non-Optional fields: known before pipeline starts (ticket, customer_id, retry_count)
- Optional fields: filled progressively as each agent runs
- No Annotated/add_messages: all fields are simple overwrites, not accumulations
"""

from typing import TypedDict, Optional


class SupportIQState(TypedDict):

    # --- Pipeline inputs (known at START) ---
    ticket: str                          # original customer ticket text
    customer_id: str                     # used to look up customer memory

    # --- Agent 1: Intent Classifier ---
    intent: Optional[str]                # refund/complaint/inquiry/technical/escalation
    priority: Optional[str]             # low/medium/high/critical

    # --- Agent 2: RAG Retrieval ---
    retrieved_chunks: Optional[list]     # top 3 chunks from hybrid_search

    # --- Agent 3: Response Generator ---
    draft_response: Optional[str]        # LLM-generated draft response
    retry_count: int                     # tracks retries — starts at 0, max 2

    # --- Agent 4: Quality Evaluator ---
    quality_scores: Optional[dict]       # faithfulness, relevance scores
    quality_passed: Optional[bool]       # True = PASS → HITL, False = BLOCK → retry
    rejection_reason: Optional[str]      # why blocked — fed back to Agent 3 on retry

    # --- Agent 5: HITL Review ---
    final_response: Optional[str]        # approved or edited response
    hitl_decision: Optional[str]         # approve / edit / reject