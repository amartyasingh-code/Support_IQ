"""
agents/quality_evaluator.py

Agent 4 in the SupportIQ pipeline.
Evaluates draft responses using DeepEval with Groq as judge LLM.

Two metrics:
    Faithfulness    — every claim grounded in retrieved chunks?
    Answer Relevance — response actually addresses the ticket?

Routing (decided in graph.py, not here):
    Both pass  → HITL Review (Agent 5)
    Either fail → retry Agent 3 (max 2 retries)
    Retries exhausted → escalate to Agent 5 directly
"""

from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from config import GROQ_API_KEY
from state import SupportIQState

# --- Quality thresholds ---
FAITHFULNESS_THRESHOLD = 0.5
RELEVANCE_THRESHOLD = 0.5


class GroqJudge(DeepEvalBaseLLM):
    """
    Wraps Groq's Llama model as a DeepEval judge LLM.
    Follows DeepEval's official custom model pattern exactly.
    """

    def __init__(self):
        self.model = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.0,
            api_key=GROQ_API_KEY,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return "groq/openai/gpt-oss-120b"

# --- Eager load judge ---
print("Loading DeepEval judge (Groq gpt-oss-120b)...")
_judge = GroqJudge()
print("DeepEval judge ready.")


def evaluate_response(
    ticket: str,
    retrieved_chunks: list,
    draft_response: str,
) -> dict:
    """
    Runs DeepEval faithfulness and relevance evaluation.
    Returns scores, pass/fail decision, and rejection reason.
    """
    context_strings = [
        chunk.get("content", "")[:400]
        for chunk in retrieved_chunks
        if chunk.get("source") != "system"
    ]

    if not context_strings:
        return {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "passed": False,
            "reason": "No policy documents retrieved — "
                      "cannot evaluate without source material."
        }

    test_case = LLMTestCase(
        input=ticket,
        actual_output=draft_response,
        retrieval_context=context_strings,
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=_judge,
        include_reason=True,
    )
    relevance_metric = AnswerRelevancyMetric(
        threshold=RELEVANCE_THRESHOLD,
        model=_judge,
        include_reason=True,
    )

    try:
        faithfulness_metric.measure(test_case)
        relevance_metric.measure(test_case)

        faithfulness_score = faithfulness_metric.score
        relevance_score = relevance_metric.score

        passed = (
            faithfulness_score >= FAITHFULNESS_THRESHOLD
            and relevance_score >= RELEVANCE_THRESHOLD
        )

        reason_parts = []
        if faithfulness_score < FAITHFULNESS_THRESHOLD:
            reason_parts.append(
                f"Faithfulness too low ({faithfulness_score:.2f}): "
                f"{faithfulness_metric.reason}"
            )
        if relevance_score < RELEVANCE_THRESHOLD:
            reason_parts.append(
                f"Relevance too low ({relevance_score:.2f}): "
                f"{relevance_metric.reason}"
            )

        return {
            "faithfulness": round(faithfulness_score, 3),
            "relevance": round(relevance_score, 3),
            "passed": passed,
            "reason": " | ".join(reason_parts) if reason_parts else None,
        }

    except Exception as e:
        print(f"Quality Evaluator → evaluation failed ({e}), defaulting to PASS")
        return {
            "faithfulness": -1.0,
            "relevance": -1.0,
            "passed": True,
            "reason": f"Evaluation failed: {e}. Human reviewer should "
                      f"assess quality manually."
        }


def quality_evaluator_node(state: SupportIQState) -> dict:
    """
    Agent 4: Evaluates draft response quality.

    Reads from state:  ticket, retrieved_chunks, draft_response, retry_count
    Writes to state:   quality_scores, quality_passed, rejection_reason
    """
    ticket = state["ticket"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    draft_response = state.get("draft_response", "")
    retry_count = state.get("retry_count", 0)

    if not draft_response or not draft_response.strip():
        print("Quality Evaluator → empty draft, blocking")
        return {
            "quality_scores": {"faithfulness": 0.0, "relevance": 0.0},
            "quality_passed": False,
            "rejection_reason": "Draft response is empty."
        }

    print(f"Quality Evaluator → evaluating (attempt {retry_count + 1})...")
    scores = evaluate_response(ticket, retrieved_chunks, draft_response)

    print(f"Quality Evaluator → faithfulness: {scores['faithfulness']}, "
          f"relevance: {scores['relevance']}, passed: {scores['passed']}")

    if scores["passed"]:
        print("Quality Evaluator → PASS → proceeding to HITL review")
    else:
        print(f"Quality Evaluator → BLOCK → {scores['reason']}")

    return {
        "quality_scores": {
            "faithfulness": scores["faithfulness"],
            "relevance": scores["relevance"],
        },
        "quality_passed": scores["passed"],
        "rejection_reason": scores["reason"],
    }


if __name__ == "__main__":
    from memory import init_db
    init_db()

    # Test with a good response — should PASS
    test_ticket = "What is my liability if I report fraud after 5 days?"

    test_chunks = [
        {
            "content": "5. LIABILITY FRAMEWORK\n"
                      "5.2 Limited liability when reported 4-7 days: "
                      "maximum INR 10,000.",
            "source": "data\\policies\\08_fraud_claim_escalation.txt",
        }
    ]

    good_response = (
        "Dear Amit Singh, according to NovaPay's Fraud Claim Policy "
        "Section 5.2, if you report fraud within 4-7 days, your liability "
        "is limited to a maximum of INR 10,000. Please contact our fraud "
        "team immediately to initiate the process."
    )

    print(f"Ticket: {test_ticket}")
    print(f"Response: {good_response[:100]}...")
    print("-" * 60)

    result = evaluate_response(test_ticket, test_chunks, good_response)
    print(f"Faithfulness: {result['faithfulness']}")
    print(f"Relevance:    {result['relevance']}")
    print(f"Passed:       {result['passed']}")
    print(f"Reason:       {result['reason']}")
