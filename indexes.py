"""
indexes.py
----------
Manages building, persisting, and loading the dual RAG indexes.

Two indexes are created from the same enriched node set:
  - VectorStoreIndex  : Fast semantic search for specific facts/details.
  - SummaryIndex      : Tree-based summarization for high-level overviews.

Both indexes share a SINGLE StorageContext so they are saved together
and can be loaded reliably from the same ./storage directory.

Bug fixed from original: previously only vector_index.storage_context.persist()
was called, which left summary_index data unsaved.
"""

import os

from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
)

from config import PERSIST_DIR, VECTOR_INDEX_ID, SUMMARY_INDEX_ID
from ingestion import load_and_ingest_documents


def build_and_persist_indexes() -> tuple[VectorStoreIndex, SummaryIndex]:
    """
    Runs the ingestion pipeline and builds both indexes from scratch.
    Both indexes share one StorageContext and are saved together.

    Returns:
        Tuple of (VectorStoreIndex, SummaryIndex)
    """
    print("🚀 No existing storage found. Starting fresh ingestion...")

    nodes = load_and_ingest_documents()

    # Create a SHARED storage context so both indexes are saved together
    storage_context = StorageContext.from_defaults()

    vector_index  = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    summary_index = SummaryIndex(nodes=nodes, storage_context=storage_context)

    # Assign deterministic IDs so we can retrieve them by ID later
    vector_index.set_index_id(VECTOR_INDEX_ID)
    summary_index.set_index_id(SUMMARY_INDEX_ID)

    # Persist ONCE via the shared context — saves both indexes
    storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"💾 Indexes saved to '{PERSIST_DIR}'")

    return vector_index, summary_index


def load_indexes_from_disk() -> tuple[VectorStoreIndex, SummaryIndex]:
    """
    Loads both indexes from the persisted ./storage directory.

    Returns:
        Tuple of (VectorStoreIndex, SummaryIndex)
    """
    print(f"📂 Loading existing indexes from '{PERSIST_DIR}'...")

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)

    vector_index  = load_index_from_storage(storage_context, index_id=VECTOR_INDEX_ID)
    summary_index = load_index_from_storage(storage_context, index_id=SUMMARY_INDEX_ID)

    print("✅ Indexes loaded successfully.")
    return vector_index, summary_index


def get_indexes() -> tuple[VectorStoreIndex, SummaryIndex]:
    """
    Entry point: returns indexes from disk if available, otherwise builds them.

    Returns:
        Tuple of (VectorStoreIndex, SummaryIndex)
    """
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return load_indexes_from_disk()
    else:
        return build_and_persist_indexes()
