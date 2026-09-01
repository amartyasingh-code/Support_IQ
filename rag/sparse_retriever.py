"""
rag/sparse_retriever.py

Sparse retrieval using BM25 keyword search with spaCy lemmatization.
Complements dense retrieval by matching exact terms, policy codes,
and banking-specific terminology that semantic search misses.

Part of the hybrid retrieval pipeline:
    dense_retriever.py  ← Chroma vector similarity search
    sparse_retriever.py ← you are here
    hybrid.py           ← RRF fusion of both results
"""

import re
import numpy as np
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi
import spacy

# --- Constants (must match ingest.py exactly) ---
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "novapay_policies"
TOP_K = 5

# --- Load spaCy model once at module level ---
print("Loading spaCy model...")
_nlp = spacy.load("en_core_web_sm")


def tokenize(text: str) -> list[str]:
    """
    Converts text to lowercase lemmatized tokens with
    stopwords and punctuation removed.

    Applied identically to both document chunks (at index
    build time) and queries (at search time).

    Example:
        "What is my liability if I report fraud after 5 days"
        → ["liability", "report", "fraud", "5", "day"]
    """
    doc = _nlp(text.lower())
    return [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.lemma_.strip()
    ]


# --- Load chunks from Chroma and build BM25 index ---
print("Building BM25 index from Chroma chunks...")

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_collection(name=COLLECTION_NAME)
_all_chunks = _collection.get(include=["documents", "metadatas"])

_tokenized_chunks = [tokenize(doc) for doc in _all_chunks["documents"]]
_bm25_index = BM25Okapi(_tokenized_chunks)

print(f"Sparse retriever ready — {len(_tokenized_chunks)} chunks indexed.")


def sparse_search(query: str, k: int = TOP_K) -> list[dict]:
    """
    Searches the BM25 index for chunks with the highest
    keyword match scores for the given query.

    Args:
        query: Customer ticket text to search against
        k: Number of top chunks to return (default: TOP_K=5)

    Returns:
        List of dicts, each containing:
            - content: the chunk text
            - source: which policy document it came from
            - score: BM25 relevance score (higher = more relevant)
            - retriever: always "sparse" (used by hybrid.py for fusion)
    """
    tokenized_query = tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:k]

    formatted = []
    for idx in top_indices:
        score = scores[idx]

        if score == 0.0:
            continue

        formatted.append({
            "content": _all_chunks["documents"][idx],
            "source": _all_chunks["metadatas"][idx].get("source", "unknown"),
            "score": round(float(score), 4),
            "retriever": "sparse",
        })

    return formatted


if __name__ == "__main__":
    test_query = "What is my liability if I report fraud after 5 days?"
    print(f"\nTest query: {test_query}\n")

    results = sparse_search(test_query)

    if not results:
        print("No results found — check tokenization or BM25 index.")
    else:
        for i, r in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Source : {r['source']}")
            print(f"  Score  : {r['score']}")
            print(f"  Content: {r['content'][:150]}...")
            print()