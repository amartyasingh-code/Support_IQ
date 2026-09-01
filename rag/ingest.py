"""
rag/ingest.py

One-time setup script: loads NovaPay policy documents, chunks them,
embeds them using sentence-transformers, and stores in a persistent
Chroma vector database.

Run this ONCE before starting the main application:
    python rag/ingest.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from langchain_community.document_loaders import TextLoader
except ImportError:
    from langchain_core.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# --- Paths ---
POLICIES_DIR = Path("data/policies")
CHROMA_DIR = Path("chroma_db")

# --- Chunking parameters ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Embedding model ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- Chroma collection name ---
COLLECTION_NAME = "novapay_policies"


def load_policy_documents():
    """
    Loads all .txt policy files from the policies directory.
    Returns a list of LangChain Document objects.
    """
    documents = []
    policy_files = sorted(POLICIES_DIR.glob("*.txt"))

    if not policy_files:
        raise FileNotFoundError(
            f"No .txt files found in {POLICIES_DIR}. "
            "Make sure policy documents are in data/policies/"
        )

    print(f"Found {len(policy_files)} policy documents:")
    for file_path in policy_files:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        documents.extend(docs)
        print(f"  Loaded: {file_path.name}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents):
    """
    Splits loaded documents into overlapping chunks using
    RecursiveCharacterTextSplitter.
    Returns a list of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunking complete:")
    print(f"  Total chunks created: {len(chunks)}")
    print(f"  Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    return chunks


def embed_and_store(chunks):
    """
    Embeds chunks using all-mpnet-base-v2 and stores them
    in a persistent Chroma vector database.
    """
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads ~420MB — this is normal)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("\nEmbedding chunks and storing in Chroma...")
    print("This may take 1-2 minutes on first run...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"\nChroma store created successfully:")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Location: {CHROMA_DIR}")
    print(f"  Total vectors stored: {vectorstore._collection.count()}")

    return vectorstore


def main():
    print("=" * 50)
    print("SupportIQ — NovaPay Policy Ingestion")
    print("=" * 50)

    # Check if Chroma already exists
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"\nChroma database already exists at {CHROMA_DIR}")
        response = input("Re-ingest and overwrite? (y/n): ").strip().lower()
        if response != "y":
            print("Skipping ingestion. Existing database kept.")
            return

    # Run pipeline
    documents = load_policy_documents()
    chunks = chunk_documents(documents)
    embed_and_store(chunks)

    print("\nIngestion complete. Ready to run SupportIQ.")
    print("=" * 50)


if __name__ == "__main__":
    main()