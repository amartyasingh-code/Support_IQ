"""
main.py

CLI entry point for SupportIQ.
Alternative to the Streamlit UI — useful for quick testing,
scripting, or environments without a browser.

Usage:
    python main.py "your ticket text" CUST-001
    python main.py  (prompts interactively)
"""

import sys
from graph import run_pipeline
from memory import init_db


def check_env():
    """Verify environment is ready before running the pipeline."""
    try:
        import config  # triggers GROQ_API_KEY validation in config.py
    except EnvironmentError as e:
        print(f"Environment check failed: {e}")
        sys.exit(1)


def print_result(result: dict):
    """Formats and prints the pipeline result to the terminal."""
    print("\n" + "=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)
    print(f"Customer ID:    {result.get('customer_id')}")
    print(f"Intent:         {result.get('intent')}")
    print(f"Priority:       {result.get('priority')}")
    print(f"Retry Count:    {result.get('retry_count')}")

    scores = result.get("quality_scores", {})
    faithfulness = scores.get("faithfulness", -1)
    relevance = scores.get("relevance", -1)

    if faithfulness == -1:
        print("Faithfulness:   Not evaluated (rate limit or error)")
    else:
        print(f"Faithfulness:   {faithfulness}")

    if relevance == -1:
        print("Relevance:      Not evaluated (rate limit or error)")
    else:
        print(f"Relevance:      {relevance}")

    if result.get("escalated"):
        print("\n🚨 ESCALATED — failed quality checks twice, needs priority review")

    print("\n" + "-" * 60)
    print("DRAFT RESPONSE:")
    print("-" * 60)
    print(result.get("draft_response", "No response generated."))
    print("=" * 60)


def main():
    check_env()
    init_db()

    if len(sys.argv) >= 3:
        ticket = sys.argv[1]
        customer_id = sys.argv[2]
    else:
        ticket = input("Enter customer ticket: ").strip()
        customer_id = input("Enter customer ID (e.g. CUST-001, or NEW-CUSTOMER): ").strip()

    if not ticket:
        print("No ticket provided. Exiting.")
        sys.exit(1)

    if not customer_id:
        customer_id = "NEW-CUSTOMER"

    print(f"\nProcessing ticket for {customer_id}...")
    print("This may take 20-40 seconds.\n")

    try:
        result = run_pipeline(ticket, customer_id)
        print_result(result)
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()