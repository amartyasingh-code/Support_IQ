"""
rag/hybrid.py

Hybrid retrieval using Reciprocal Rank Fusion (RRF).
Combines dense (Chroma) and sparse (BM25) results into
one unified ranking — no raw score comparison needed.

Why RRF instead of score averaging:
Dense scores (0-1) and BM25 scores (0-10+) are on completely
different scales. Direct addition means BM25 dominates 97% of
the combined score. RRF works on rank positions instead —
both retrievers contribute equally regardless of score scale.

Part of the hybrid retrieval pipeline:
    dense_retriever.py  ← Chroma vector similarity search
    sparse_retriever.py ← BM25 keyword search
    hybrid.py           ← you are here
"""

from rag.dense_retriever import dense_search
from rag.sparse_retriever import sparse_search

# --- RRF smoothing constant ---
# k=60 is the empirically proven default from the original
# RRF paper (Cormack, Clarke, Buettcher 2009)
RRF_K = 60

# --- Final number of chunks to return to the LLM ---
# 5 dense + 5 sparse = 10 candidates → RRF picks best 3
FINAL_TOP_K = 2


def hybrid_search(query: str, final_k: int = FINAL_TOP_K) -> list[dict]:
    """
    Combines dense and sparse retrieval results using
    Reciprocal Rank Fusion (RRF).

    Args:
        query: Customer ticket text to search against
        final_k: Number of final chunks to return (default: 3)

    Returns:
        List of dicts, each containing:
            - content: chunk text
            - source: policy document it came from
            - rrf_score: combined RRF score
            - retrievers: which retrievers found this chunk
    """
    # Step 1: get results from both retrievers
    dense_results = dense_search(query)
    sparse_results = sparse_search(query)

    # Step 2: accumulate RRF scores keyed by chunk content
    rrf_scores = {}

    # Process dense results
    for rank, result in enumerate(dense_results, start=1):
        content = result["content"]
        rrf_contribution = 1 / (RRF_K + rank)

        if content not in rrf_scores:
            rrf_scores[content] = {
                "content": content,
                "source": result["source"],
                "rrf_score": 0.0,
                "retrievers": [],
            }

        rrf_scores[content]["rrf_score"] += rrf_contribution
        rrf_scores[content]["retrievers"].append("dense")

    # Process sparse results
    for rank, result in enumerate(sparse_results, start=1):
        content = result["content"]
        rrf_contribution = 1 / (RRF_K + rank)

        if content not in rrf_scores:
            rrf_scores[content] = {
                "content": content,
                "source": result["source"],
                "rrf_score": 0.0,
                "retrievers": [],
            }

        rrf_scores[content]["rrf_score"] += rrf_contribution
        rrf_scores[content]["retrievers"].append("sparse")

    # Step 3: sort by RRF score descending
    ranked = sorted(
        rrf_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    # Step 4: round scores and return top final_k
    for result in ranked:
        result["rrf_score"] = round(result["rrf_score"], 6)

    return ranked[:final_k]


if __name__ == "__main__":
    test_query = "What is my liability if I report fraud after 5 days?"
    print(f"\nTest query: {test_query}\n")

    results = hybrid_search(test_query)

    print(f"Top {len(results)} chunks after RRF fusion:\n")
    for i, r in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Source    : {r['source']}")
        print(f"  RRF Score : {r['rrf_score']}")
        print(f"  Retrievers: {r['retrievers']}")
        print(f"  Content   : {r['content'][:150]}...")
        print()