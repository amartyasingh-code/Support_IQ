"""
agents/response_generator.py

Agent 3 in the SupportIQ pipeline.
Generates a grounded, personalized response using:
    - Retrieved policy chunks (from Agent 2)
    - Customer interaction history (from memory.py)
    - Rejection reason if this is a retry (from Agent 4)

Grounding principle: response must only use information from
retrieved chunks — no LLM general knowledge about banking policies.
"""

import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import GROQ_API_KEY
from state import SupportIQState
from memory import get_customer_history

# --- LLM setup ---
_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=GROQ_API_KEY,
)

# --- Policy document name mapping ---
_SOURCE_NAMES = {
    "01_savings_account_terms": "NovaPay Savings Account Terms",
    "02_credit_card_refund_policy": "NovaPay Credit Card Refund Policy",
    "03_upi_dispute_resolution": "NovaPay UPI Dispute Resolution Policy",
    "04_kyc_requirements": "NovaPay KYC Requirements",
    "05_loan_eligibility": "NovaPay Loan Eligibility Policy",
    "06_account_freeze_unfreeze": "NovaPay Account Freeze Policy",
    "07_fixed_deposit_policy": "NovaPay Fixed Deposit Policy",
    "08_fraud_claim_escalation": "NovaPay Fraud Claim and Escalation Policy",
    "08_fraud_claim_escalation.txt": "NovaPay Fraud Claim and Escalation Policy",
    "09_grievance_redressal": "NovaPay Grievance Redressal Policy",
    "10_internet_mobile_banking_security": "NovaPay Internet Banking Security Policy",
}


def _format_chunks(chunks: list) -> str:
    """
    Converts retrieved chunks into formatted string for LLM prompt.
    Cleans file paths into readable policy document names.
    """
    if not chunks:
        return "No relevant policy documents found."

    formatted = []
    for chunk in chunks:
        raw_source = chunk.get("source", "unknown")
        filename = Path(raw_source).stem
        clean_name = _SOURCE_NAMES.get(filename, filename)

        formatted.append(
            f"SOURCE: {clean_name}\n"
            f"CONTENT:\n{chunk['content']}"
        )

    return "\n\n---\n\n".join(formatted)


def _format_customer_history(history: dict) -> str:
    """
    Converts customer history dict into readable string for LLM prompt.
    Includes only what the LLM needs to personalize the response.
    """
    if not history["found"]:
        return "This is a first-time customer — no previous interaction history."

    lines = [
        f"Customer Name: {history['name']}",
        f"Sentiment: {history['sentiment']}",
        f"Total previous tickets: {history['total_tickets']}",
    ]

    if history["recent_tickets"]:
        lines.append("Recent ticket history:")
        for i, ticket in enumerate(history["recent_tickets"], 1):
            lines.append(
                f"  {i}. [{ticket['intent'].upper()}] "
                f"{ticket['ticket_text'][:80]}... "
                f"→ Resolution: {ticket['resolution']}"
            )
    else:
        lines.append("No previous tickets on record.")

    return "\n".join(lines)


# --- System prompt ---
RESPONSE_GENERATOR_PROMPT = """You are a professional customer support agent \
for NovaPay Digital Bank. Your job is to write accurate, empathetic, and \
grounded responses to customer support tickets.

GROUNDING RULES (most important):
- Base your response ONLY on the policy chunks provided below
- Do NOT use general banking knowledge or make up policy details
- If the provided chunks do not contain enough information to answer \
the question, say so clearly — do not invent an answer
- Cite the policy source when making specific claims \
(e.g. "According to NovaPay's Fraud Claim Policy...")

TONE RULES:
- neutral/positive sentiment → professional and friendly tone
- frustrated sentiment → empathetic, acknowledge difficulty, \
extra reassurance that the issue will be resolved
- angry sentiment → apologetic first, acknowledge NovaPay's \
failure to resolve previous issues, concrete next steps

RESPONSE FORMAT:
- Start with a personalized greeting using the customer's name
- Address their specific question directly — no generic preamble
- Cite relevant policy sections to support your answer
- End with a clear next step or resolution timeline
- Keep response between 150-250 words — concise but complete
- Professional banking language — no slang, no casual abbreviations

DO NOT:
- Apologize excessively (once is enough)
- Make promises about timelines you cannot guarantee
- Share information from one customer's history with another
- Use phrases like "As an AI" or "I am a language model"
- Sign off as "[Your Name]" — always sign as "NovaPay Support Team"
"""


def response_generator_node(state: SupportIQState) -> dict:
    """
    Agent 3: Generates a grounded, personalized response.

    Reads from state:  ticket, customer_id, retrieved_chunks,
                       intent, priority, retry_count, rejection_reason
    Writes to state:   draft_response, retry_count

    Returns partial state update.
    """
    ticket = state["ticket"]
    customer_id = state["customer_id"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    intent = state.get("intent", "inquiry")
    priority = state.get("priority", "medium")
    retry_count = state.get("retry_count", 0)
    rejection_reason = state.get("rejection_reason", None)

    customer_history = get_customer_history(customer_id)
    formatted_history = _format_customer_history(customer_history)
    formatted_chunks = _format_chunks(retrieved_chunks)

    if retry_count == 0 or not rejection_reason:
        task_instruction = f"""Please write a response to this customer ticket.

CUSTOMER INFORMATION:
{formatted_history}

TICKET DETAILS:
Intent: {intent}
Priority: {priority}
Ticket: {ticket}

RELEVANT POLICY SECTIONS:
{formatted_chunks}"""

    else:
        task_instruction = f"""Your previous response was rejected by the \
quality evaluator for the following reason:

REJECTION REASON: {rejection_reason}

Please rewrite the response addressing these specific issues.

CUSTOMER INFORMATION:
{formatted_history}

TICKET DETAILS:
Intent: {intent}
Priority: {priority}
Ticket: {ticket}

RELEVANT POLICY SECTIONS:
{formatted_chunks}"""

    messages = [
        SystemMessage(content=RESPONSE_GENERATOR_PROMPT),
        HumanMessage(content=task_instruction),
    ]

    try:
        response = _llm.invoke(messages)
        draft = response.content.strip()

        print(f"Response Generator → generated {len(draft.split())} word "
              f"response (attempt {retry_count + 1})")

        new_retry_count = retry_count + 1 if rejection_reason else retry_count

        return {
            "draft_response": draft,
            "retry_count": new_retry_count,
        }

    except Exception as e:
        print(f"Response Generator → failed ({e})")
        fallback = (
            f"Dear {customer_history.get('name', 'Valued Customer')}, "
            f"thank you for contacting NovaPay support. We have received "
            f"your query and a support agent will contact you within 24 hours. "
            f"Reference: {customer_id}"
        )
        new_retry_count = retry_count + 1 if rejection_reason else retry_count

        return {
            "draft_response": fallback,
            "retry_count": new_retry_count,
        }


if __name__ == "__main__":
    from state import SupportIQState
    from memory import init_db
    init_db()

    test_state = SupportIQState(
        ticket="What is my liability if I report fraud after 5 days?",
        customer_id="CUST-003",
        retry_count=0,
        intent="technical",
        priority="critical",
        retrieved_chunks=[
            {
                "content": "5. LIABILITY FRAMEWORK (AS PER RBI CIRCULAR RBI/2017-18/15)\n"
                          "5.1 Zero customer liability when fraud due to NovaPay negligence.\n"
                          "5.2 Limited liability when reported 4-7 days: max INR 10,000.\n"
                          "5.3 Full liability when reported after 7 days.",
                "source": "data\\policies\\08_fraud_claim_escalation.txt",
                "rrf_score": 0.032,
                "retrievers": ["dense", "sparse"]
            },
            {
                "content": "4. ZERO LIABILITY PROTECTION\n"
                          "4.1 Customers not liable for unauthorized transactions IF:\n"
                          "    - Card reported lost before transaction\n"
                          "    - Transaction due to NovaPay system error\n"
                          "    - Dispute reported within 3 days",
                "source": "data\\policies\\02_credit_card_refund_policy.txt",
                "rrf_score": 0.031,
                "retrievers": ["dense", "sparse"]
            },
        ],
        quality_scores=None,
        quality_passed=None,
        rejection_reason=None,
        draft_response=None,
        final_response=None,
        hitl_decision=None,
        escalated=None,
    )

    print("Testing Response Generator — first attempt...")
    print(f"Customer: CUST-003 (Amit Singh — angry, 3 previous tickets)")
    print(f"Ticket: {test_state['ticket']}")
    print("-" * 60)

    result = response_generator_node(test_state)
    print("\nDRAFT RESPONSE:")
    print("-" * 60)
    print(result["draft_response"])
    print(f"\nretry_count in result: {result.get('retry_count')}")
    print("-" * 60)