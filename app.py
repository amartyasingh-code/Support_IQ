"""
app.py

Streamlit web interface for SupportIQ.

Two pages via sidebar navigation:
    1. New Ticket — submission + HITL review flow
    2. Ticket History — dashboard of all processed tickets (SQLite)

Two-phase flow on "New Ticket":
    1. Ticket submission — customer enters ticket + customer ID, runs pipeline
    2. HITL Review — human reviewer sees full context, approves/edits/rejects

Uses st.session_state to cache pipeline results across Streamlit's
automatic script reruns — the expensive 5-agent pipeline only runs
once per ticket, not on every button click.
"""

import streamlit as st
from graph import run_pipeline
from memory import init_db, save_ticket, get_customer_history, get_all_tickets

# --- Initialize database on app startup ---
init_db()

# --- Page config ---
st.set_page_config(
    page_title="SupportIQ — NovaPay AI Support",
    page_icon="🏦",
    layout="wide",
)

# --- Session state initialization ---
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "decision_made" not in st.session_state:
    st.session_state.decision_made = False

if "edited_response" not in st.session_state:
    st.session_state.edited_response = ""

# --- Sidebar navigation ---
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Go to:", ["New Ticket", "Ticket History"])

# --- Header (shown on both pages) ---
st.title("🏦 SupportIQ")
st.markdown("**NovaPay Digital Bank** — AI-Powered Customer Support with Human Review")

# ============================================================
# PAGE 1: New Ticket
# ============================================================
if page == "New Ticket":

    st.markdown("""
    Multi-agent pipeline: 
    🎯 Intent Classification → 📚 Hybrid RAG Retrieval → ✍️ Response Generation 
    → ✅ Quality Evaluation → 👨‍💼 Human Review
    """)
    st.divider()

    # --- Screen 1: Ticket submission ---
    if st.session_state.pipeline_result is None:

        col1, col2 = st.columns([2, 1])

        with col1:
            ticket_text = st.text_area(
                "Customer Ticket",
                placeholder="e.g. What is my liability if I report fraud after 5 days?",
                height=120,
            )

        with col2:
            customer_id = st.selectbox(
                "Customer ID (test data)",
                options=["CUST-001", "CUST-002", "CUST-003", "NEW-CUSTOMER"],
                help="CUST-001: Rahul (moderate history) | "
                     "CUST-002: Priya (new, low priority) | "
                     "CUST-003: Amit (angry, 3 tickets) | "
                     "NEW-CUSTOMER: first-time contact"
            )

        run_button = st.button("🚀 Run SupportIQ Pipeline", type="primary", use_container_width=True)

        if run_button:
            if not ticket_text.strip():
                st.warning("Please enter a ticket before running the pipeline.")
            else:
                with st.spinner("Running 5-agent pipeline... this may take 20-40 seconds"):
                    try:
                        result = run_pipeline(ticket_text, customer_id)
                        st.session_state.pipeline_result = result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")

    # --- Screen 2: HITL Review ---
    elif st.session_state.pipeline_result is not None and not st.session_state.decision_made:

        result = st.session_state.pipeline_result

        if result.get("escalated"):
            st.error(
                "🚨 **ESCALATED** — This response failed automated quality "
                "checks twice. Please review carefully before approving."
            )

        st.subheader(f"📋 Reviewing Ticket for {result.get('customer_id')}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Intent", result.get("intent", "N/A"))
        with col2:
            st.metric("Priority", result.get("priority", "N/A"))
        with col3:
            st.metric("Retry Count", result.get("retry_count", 0))

        st.markdown(f"**Original Ticket:** {result.get('ticket')}")

        st.divider()

        st.subheader("📊 Quality Scores")
        scores = result.get("quality_scores", {})
        faithfulness = scores.get("faithfulness", -1)
        relevance = scores.get("relevance", -1)

        score_col1, score_col2 = st.columns(2)

        with score_col1:
            if faithfulness == -1:
                st.warning("Faithfulness: Not evaluated (rate limit or error)")
            elif faithfulness >= 0.5:
                st.success(f"✅ Faithfulness: {faithfulness}")
            else:
                st.error(f"❌ Faithfulness: {faithfulness}")

        with score_col2:
            if relevance == -1:
                st.warning("Relevance: Not evaluated (rate limit or error)")
            elif relevance >= 0.5:
                st.success(f"✅ Relevance: {relevance}")
            else:
                st.error(f"❌ Relevance: {relevance}")

        st.divider()

        with st.expander("📚 Retrieved Policy Chunks (source material)"):
            chunks = result.get("retrieved_chunks", [])
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}** — Source: `{chunk.get('source', 'unknown')}`")
                st.text(chunk.get("content", "")[:300] + "...")
                st.caption(f"Found by: {chunk.get('retrievers', ['unknown'])}")
                st.markdown("---")

        st.subheader("✍️ Draft Response")
        st.session_state.edited_response = st.text_area(
            "Review and edit if needed:",
            value=result.get("draft_response", ""),
            height=250,
        )

        st.divider()

        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            approve = st.button("✅ Approve", type="primary", use_container_width=True)
        with btn_col2:
            edit_approve = st.button("✏️ Approve Edited Version", use_container_width=True)
        with btn_col3:
            reject = st.button("❌ Reject", use_container_width=True)

        if approve or edit_approve:
            final_text = result.get("draft_response") if approve else st.session_state.edited_response
            decision = "approved" if approve else "approved_edited"

            customer_history = get_customer_history(result.get("customer_id"))
            actual_name = customer_history.get("name", result.get("customer_id"))

            save_ticket(
                customer_id=result.get("customer_id"),
                name=actual_name,
                ticket_text=result.get("ticket"),
                intent=result.get("intent"),
                priority=result.get("priority"),
                resolution=final_text[:200],
            )

            st.session_state.final_response = final_text
            st.session_state.hitl_decision = decision
            st.session_state.decision_made = True
            st.rerun()

        if reject:
            st.session_state.hitl_decision = "rejected"
            st.session_state.decision_made = True
            st.rerun()

    # --- Screen 3: Completion ---
    elif st.session_state.decision_made:

        decision = st.session_state.get("hitl_decision", "unknown")

        if decision in ("approved", "approved_edited"):
            st.success("✅ Response approved and sent to customer!")
            st.markdown("**Final Response Sent:**")
            st.info(st.session_state.get("final_response", ""))
        else:
            st.error("❌ Response rejected. This ticket requires manual handling.")

        st.divider()

        if st.button("🔄 Process New Ticket"):
            st.session_state.pipeline_result = None
            st.session_state.decision_made = False
            st.session_state.edited_response = ""
            st.rerun()

# ============================================================
# PAGE 2: Ticket History
# ============================================================
elif page == "Ticket History":

    st.subheader("📜 All Processed Tickets")

    tickets = get_all_tickets()

    if not tickets:
        st.info("No tickets processed yet. Go to 'New Ticket' to process one.")
    else:
        st.markdown(f"**Total tickets processed:** {len(tickets)}")
        st.divider()

        for t in tickets:
            intent_label = t["intent"].upper() if t["intent"] else "N/A"
            customer_label = t["name"] or t["customer_id"]
            preview = t["ticket_text"][:60] if t["ticket_text"] else ""

            with st.expander(f"[{intent_label}] {customer_label} — {preview}..."):
                st.markdown(f"**Customer:** {t['name']} ({t['customer_id']})")
                st.markdown(f"**Priority:** {t['priority']}")
                st.markdown(f"**Ticket:** {t['ticket_text']}")
                st.markdown(f"**Resolution:** {t['resolution']}")
                st.caption(f"Processed: {t['created_at']}")