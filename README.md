# 🏦 SupportIQ — AI Customer Support Platform

A production-grade, multi-agent AI system that automates the complete customer support workflow for **NovaPay Digital Bank** (fictional) — from ticket intake to human-approved response delivery.

> Built as a portfolio project to demonstrate multi-agent orchestration, hybrid RAG retrieval, LLM-as-judge evaluation, and human-in-the-loop design for LLM Engineer / AI Agents roles.

---

## 📌 What It Does

A customer submits a support ticket. Five specialized AI agents collaborate to:
1. Classify the ticket's intent and priority
2. Retrieve relevant NovaPay policy sections using hybrid search
3. Draft a grounded, personalized response using policy content + customer history
4. Evaluate the response for faithfulness and relevance — auto-retry if it fails
5. Present the final package to a human reviewer for approval, editing, or rejection

No response reaches a customer without either passing automated quality checks or being explicitly approved by a human.

---

## 🏗️ Architecture

```
Customer Ticket (PII masked)
         │
         ▼
┌─────────────────────┐
│ 1. Intent Classifier │  Pydantic-enforced structured output:
│    (Groq LLM)        │  5 intents × 4 priority levels
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ 2. RAG Retrieval     │  Hybrid search over 10 NovaPay policy docs:
│    (Hybrid RAG)       │  Dense (Chroma) + Sparse (BM25) + RRF fusion
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ 3. Response Generator│  Grounded, personalized draft using
│    (Groq LLM)         │  retrieved chunks + SQLite customer memory
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ 4. Quality Evaluator  │  DeepEval: Faithfulness + Answer Relevance
│    (Custom Groq judge)│  scoring via LLM-as-judge
└──────────┬───────────┘
           │
     ┌─────┴──────┐
   PASS          BLOCK (retry < 2)
     │              │
     │              ▼
     │      back to Response Generator
     │      (with rejection reason)
     │
     ▼
┌─────────────────────┐
│ 5. HITL Reviewer      │  Human sees ticket, chunks, scores, draft
│    (Streamlit UI)     │  → Approve / Edit / Reject
└──────────┬───────────┘
           │
           ▼
    Final Response + saved to customer memory (SQLite)
```

**Conditional retry logic:** if the Quality Evaluator blocks a response, it loops back to the Response Generator (max 2 retries) with the specific rejection reason as context. After 2 failed retries, the ticket is force-escalated to the human reviewer regardless of score, flagged with a priority warning banner.

---

## 🔍 Hybrid RAG — Proven, Not Assumed

Rather than assuming vector search alone is sufficient, retrieval quality was benchmarked directly:

**Query:** *"What is my liability if I report fraud after 5 days?"*

| Retriever | Rank of correct chunk | Score |
|---|---|---|
| Dense only (Chroma) | 2nd | 0.264 (cosine similarity) |
| Sparse only (BM25) | 1st | 10.64 (BM25 score) |
| **Hybrid (RRF fusion)** | **1st** | **0.0325 (RRF score)**, found by both retrievers |

Dense search captures semantic intent but dilutes exact banking terminology (policy codes, specific amounts, regulatory references). BM25 captures precise keyword matches but misses paraphrased queries. Reciprocal Rank Fusion (k=60) combines both — hand-rolled rather than using LangChain's `EnsembleRetriever`, so every fusion score is inspectable and explainable.

---

## 🛠️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) | Free tier, fast inference |
| Orchestration | LangGraph `StateGraph` | Stateful multi-agent flow with conditional routing |
| Dense retrieval | ChromaDB + `all-mpnet-base-v2` | Local, free, CPU-based semantic search |
| Sparse retrieval | `rank_bm25` + spaCy | Keyword matching with lemmatization (preserves banking terms like "liability" that Porter stemming mangles) |
| Fusion | Hand-rolled Reciprocal Rank Fusion | Fully inspectable, no black-box library |
| Structured output | Pydantic + `Literal` types | Guarantees valid intent/priority categories |
| RAG evaluation | DeepEval (custom Groq judge) | Faithfulness + Answer Relevance scoring |
| Memory | SQLite | Customer history, sentiment tracking, structured queries |
| PII protection | Custom regex masking | Card numbers, Aadhaar, PAN, CVV, OTP, email, phone |
| UI | Streamlit | HITL review interface + ticket history dashboard |

**Note on DeepEval vs RAGAS:** RAGAS was the original evaluation choice but was dropped due to a confirmed upstream bug (`langchain_community.chat_models.vertexai` import failure affecting all RAGAS 0.x/0.4.x versions with newer LangChain). DeepEval was evaluated as an alternative and adopted after confirming compatibility.

---

## 📁 Project Structure

```
supportiq/
├── data/
│   ├── policies/              # 10 synthetic NovaPay policy documents
│   └── customers/             # SQLite database (gitignored)
│
├── rag/
│   ├── ingest.py               # Load → chunk → embed → store (run once)
│   ├── dense_retriever.py      # Chroma vector similarity search
│   ├── sparse_retriever.py     # BM25 + spaCy lemmatization
│   └── hybrid.py                # Reciprocal Rank Fusion
│
├── agents/
│   ├── intent_classifier.py     # Agent 1 — Pydantic structured output
│   ├── rag_retrieval.py         # Agent 2 — hybrid search + query enrichment
│   ├── response_generator.py    # Agent 3 — grounded, personalized drafting
│   ├── quality_evaluator.py     # Agent 4 — DeepEval + custom Groq judge
│   └── hitl_reviewer.py         # Agent 5 — escalation detection
│
├── notebooks/
│   └── retrieval_diagnostic.ipynb   # Dense vs sparse vs hybrid comparison
│
├── state.py             # Shared LangGraph state schema (TypedDict)
├── config.py            # Centralized env loading + pipeline constants
├── memory.py            # SQLite customer history (read/write/history)
├── pii_masking.py       # Regex-based PII detection and masking
├── graph.py             # LangGraph StateGraph wiring all 5 agents
├── app.py               # Streamlit UI (submission + review + history)
├── main.py              # CLI entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## ⚙️ Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/your-username/supportiq.git
cd supportiq

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Set up your free Groq API key
cp .env.example .env
# Edit .env and add your key from https://console.groq.com/keys

# 5. Ingest the policy documents (one-time setup)
python rag/ingest.py

# 6. Run the app
streamlit run app.py
```

**Test customer IDs pre-seeded in the database:** `CUST-001` (Rahul, moderate history), `CUST-002` (Priya, new/low-priority), `CUST-003` (Amit, angry/3 critical tickets), or `NEW-CUSTOMER` for a first-time contact.

---

## 🚀 Usage

**Streamlit (recommended):**
```bash
streamlit run app.py
```
Two views via sidebar: **New Ticket** (submit + review) and **Ticket History** (dashboard of all processed tickets, joined with customer names via SQL `LEFT JOIN`).

**CLI:**
```bash
python main.py "What is my liability if I report fraud after 5 days?" CUST-003
```

---

## 🧠 Key Engineering Decisions

**Hand-rolled RRF over LangChain's `EnsembleRetriever`** — chosen deliberately to keep the fusion algorithm fully inspectable. Every RRF score can be traced back to `1/(k + rank)` contributions from each retriever, which matters both for debugging and for explaining the system in technical interviews.

**spaCy lemmatization over Porter Stemmer for BM25** — Porter mangles banking/legal terms ("liability" → "liabil", "fraudulent" → "fraudul"). spaCy's vocabulary-aware lemmatization preserves these correctly, at the cost of a larger model download.

**Query enrichment is intent-aware, not always applied** — appending intent-specific terms (e.g., "UPI NEFT IMPS transaction failure" for technical tickets) improves retrieval for vague queries but can occasionally shift ranking on already-specific queries. Documented as a measured trade-off, not treated as a universal improvement.

**Retry counter owned by the Response Generator, not the Quality Evaluator** — `retry_count` tracks "how many times the Generator has been asked to try again," so incrementing it lives with the agent whose attempts it measures. Keeps the Quality Evaluator's responsibility limited to scoring, not state management.

**DeepEval judge uses a smaller/cheaper model than the response generator** — evaluation is a comparison task, not a generation task; using a lighter model for judging reduces token consumption without sacrificing scoring quality. This was one of two changes (along with reducing retrieved chunks from 3→2 and truncating evaluation context to 400 chars) made after hitting Groq's free-tier rate limit (8000 TPM) during testing.

**PII masking runs once, at the pipeline entry point** — before the ticket reaches any LLM call (including the DeepEval judge) or SQLite storage. Regex-based rather than ML-based, chosen for zero added latency and deterministic behavior on well-defined formats (card numbers, Aadhaar, PAN, CVV, OTP).

**HITL Reviewer node only computes an `escalated` flag — Streamlit does the actual reviewing** — the LangGraph node prepares the routing signal (was this a normal PASS or a forced escalation after exhausted retries?); the interactive approve/edit/reject decision happens in the UI layer, not inside the graph. Session state in Streamlit ensures the 5-agent pipeline runs exactly once per ticket, even though Streamlit reruns the entire script on every button click.

**Python monolith over FastAPI + separate frontend** — deliberately deferred. Streamlit calls `graph.py` directly. A thin FastAPI layer wrapping `run_pipeline()` is a natural, low-risk v2 addition once the core AI architecture is proven — see Next Steps.

---

## ⚠️ Known Limitations

- **Groq free-tier rate limits** — a single ticket triggers 4-6+ LLM calls across all agents (including DeepEval's internal claim-extraction calls). Running multiple tickets in quick succession can hit the 8000 TPM limit. On failure, the Quality Evaluator defaults to `passed=True` with sentinel scores of `-1.0` (visibly flagged in the UI) rather than blocking indefinitely.
- **Relevance scores vary by response style** — thorough, helpful responses that go slightly beyond the literal question (e.g., suggesting next steps) can score lower on strict relevance than narrowly-scoped answers, even when they're more useful to a real customer. Threshold is set at 0.5, not higher, to accommodate this.
- **No live voice/call escalation** — the "escalated" UI state is a mocked warning banner, not an integration with a telephony provider. A Twilio Voice API integration is a documented next step, not implemented.
- **New customers aren't personalized by name from ticket text** — customer identification is strictly `customer_id`-driven via the database; the system does not parse names mentioned in free-text ticket content, to avoid unreliable NLP-based name extraction.
- **CLI has no interactive HITL flow** — `main.py` prints the draft response and scores for quick testing but does not support approve/edit/reject; that interaction is Streamlit-only.

---

## 🔭 Possible Next Steps

- Wrap `run_pipeline()` in a FastAPI endpoint, decoupling the AI backend from the Streamlit frontend
- Real Twilio Voice API integration for the escalation flow (currently mocked in the UI)
- Add a conditional edge allowing the Quality Evaluator to route back to the RAG Retrieval agent (not just the Response Generator) if retrieved chunks appear insufficient
- Migrate SQLite → PostgreSQL and Chroma → a managed vector store for multi-instance deployment
- Cloud migration path: Secrets Manager for API keys, RDS for customer data, ECS/Cloud Run for the FastAPI backend — deliberately out of scope for this portfolio version

---

## 👤 Author

**Amartya Singh**
System Engineer @ TCS | Transitioning into Generative AI Engineering
