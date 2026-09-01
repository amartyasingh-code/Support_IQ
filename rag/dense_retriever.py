"""
rag/dense_retriever.py

Dense retrieval using Chroma vector similarity search.
Loads the embedding model and Chroma connection once at import time
(eager loading) and reuses them for every query.

Part of the hybrid retrieval pipeline:
    dense_retriever.py  ← you are here
    sparse_retriever.py ← BM25 keyword search
    hybrid.py           ← RRF fusion of both results
"""

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Constants (must match ingest.py exactly) ---
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "novapay_policies"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 5

# --- Eager loading: runs once when module is first imported ---
print("Loading dense retriever...")

_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

_vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=str(CHROMA_DIR),
)

print(f"Dense retriever ready — {_vectorstore._collection.count()} chunks indexed.")


def dense_search(query: str, k: int = TOP_K) -> list[dict]:
    """
    Searches Chroma for the most semantically similar chunks
    to the given query.

    Args:
        query: Customer ticket text to search against
        k: Number of top chunks to return (default: TOP_K=5)

    Returns:
        List of dicts, each containing:
            - content: the chunk text
            - source: which policy document it came from
            - score: cosine similarity score (higher = more similar)
            - retriever: always "dense" (used by hybrid.py for fusion)
    """
    results = _vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=k,
    )

    formatted = []
    for doc, score in results:
        formatted.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": round(score, 4),
            "retriever": "dense",
        })

    return formatted


if __name__ == "__main__":
    test_query = "What is my liability if I report fraud after 5 days?"
    print(f"\nTest query: {test_query}\n")

    results = dense_search(test_query)

    for i, r in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Source : {r['source']}")
        print(f"  Score  : {r['score']}")
        print(f"  Content: {r['content'][:150]}...")
        print()