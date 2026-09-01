"""
agents/intent_classifier.py

Agent 1 in the SupportIQ pipeline.
Classifies incoming customer tickets by intent and priority.

Three-layer output safety:
    1. Input guardrail — empty ticket check before LLM call
    2. Prompt guardrails — instructs LLM to return only allowed values
    3. Pydantic + with_structured_output() — validates at code level
    4. Exception fallback — safe default if LLM/Pydantic fails
"""

import os
from typing import Literal
from pydantic import BaseModel
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from state import SupportIQState

load_dotenv()


class IntentOutput(BaseModel):
    """
    Pydantic schema enforcing valid intent classifier outputs.
    Literal types restrict values to exactly the allowed options.
    """
    intent: Literal[
        "refund",
        "complaint",
        "inquiry",
        "technical",
        "escalation"
    ]
    priority: Literal[
        "low",
        "medium",
        "high",
        "critical"
    ]


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- LLM: temperature 0.0 for deterministic classification ---
_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    api_key=GROQ_API_KEY,
)

# --- Bind structured output schema to LLM ---
_structured_llm = _llm.with_structured_output(IntentOutput)

# --- System prompt ---
INTENT_CLASSIFIER_PROMPT = """You are an expert customer support classifier \
for NovaPay Digital Bank.

Your job is to analyze incoming customer tickets and classify them by:
1. Intent — what the customer wants
2. Priority — how urgently this needs to be resolved

INTENT CATEGORIES:
- refund: customer wants money returned (failed transaction, wrong charge, duplicate payment)
- complaint: customer is unhappy with service, staff, or policy (no financial recovery needed)
- inquiry: customer asking a question (account info, product features, eligibility)
- technical: system/app issue (failed transaction, login problem, OTP not received)
- escalation: customer has raised this before and is frustrated with lack of resolution

PRIORITY LEVELS:
- low: general inquiry, no financial impact, not time-sensitive
- medium: service complaint, minor inconvenience, can wait 24-48 hours
- high: financial impact (money debited, transaction failed), needs same-day resolution
- critical: large amount involved, fraud suspected, account compromised, repeat escalation

RULES:
- Choose the SINGLE best matching intent — do not combine categories
- UPI/NEFT/IMPS transaction failures where money was debited but 
  transaction shows failed → ALWAYS classify as technical first, 
  not refund. The money will be auto-reversed if the transaction 
  genuinely failed. Only classify as refund if customer explicitly 
  says "I want my money back" or "please refund."
- Escalation takes priority over all other intents if the customer \
mentions previous unresolved complaints
- Critical priority if ANY of these are true:
  * A specific amount more than INR 10,000* is mentioned (e.g. INR 15,000, \
INR 50,000)
  * The words fraud, unauthorized, hacked, or compromised appear
  * Customer cannot access their account
  * Customer mentions raising the issue multiple times before
"""


def intent_classifier_node(state: SupportIQState) -> dict:
    """
    Agent 1: Classifies the customer ticket by intent and priority.

    Reads from state:  ticket
    Writes to state:   intent, priority

    Returns partial state update — only the fields this agent owns.
    """
    ticket = state["ticket"]

    # Guard 1: empty ticket check
    if not ticket or not ticket.strip():
        print("Intent Classifier → empty ticket, defaulting to inquiry/medium")
        return {"intent": "inquiry", "priority": "medium"}

    messages = [
        ("system", INTENT_CLASSIFIER_PROMPT),
        ("human", f"Classify this customer ticket:\n\n{ticket}"),
    ]

    try:
        result: IntentOutput = _structured_llm.invoke(messages)
        print(f"Intent Classifier → intent: {result.intent}, priority: {result.priority}")
        return {
            "intent": result.intent,
            "priority": result.priority,
        }

    except Exception as e:
        # Guard 2: LLM or Pydantic failure — safe fallback
        # inquiry/medium ensures ticket still flows to HITL review
        print(f"Intent Classifier → failed ({e}), defaulting to inquiry/medium")
        return {
            "intent": "inquiry",
            "priority": "medium",
        }


if __name__ == "__main__":
    test_tickets = [
        {
            "ticket": "My UPI payment of INR 15,000 failed but money was debited from my account",
            "expected": "technical / critical"
        },
        {
            "ticket": "What is the interest rate on your savings account?",
            "expected": "inquiry / low"
        },
        {
            "ticket": "I have complained 3 times about my frozen account and nobody has helped me",
            "expected": "escalation / critical"
        },
    ]

    for test in test_tickets:
        print(f"\nTicket: {test['ticket']}")
        print(f"Expected: {test['expected']}")

        state = SupportIQState(
            ticket=test["ticket"],
            customer_id="TEST-001",
            retry_count=0,
            intent=None,
            priority=None,
            retrieved_chunks=None,
            draft_response=None,
            quality_scores=None,
            quality_passed=None,
            rejection_reason=None,
            final_response=None,
            hitl_decision=None,
        )

        result = intent_classifier_node(state)
        print(f"Got: {result['intent']} / {result['priority']}")
        print("-" * 50)
