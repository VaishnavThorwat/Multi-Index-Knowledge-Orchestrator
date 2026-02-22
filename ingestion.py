"""
ingestion.py
------------
Handles document loading from ./data and runs the enrichment pipeline.

Pipeline steps:
  1. SentenceSplitter  — chunks documents into overlapping text nodes
  2. QuestionsAnsweredExtractor — enriches each node with metadata questions
     (improves retrieval quality significantly)

NOTE: Embedding is intentionally NOT part of this pipeline.
      The VectorStoreIndex handles embedding automatically when
      nodes are passed to it, avoiding double-embedding.
"""

import os
import time

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline

from config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
    BATCH_SLEEP_S,
    METADATA_QUESTIONS_PER_NODE,
    llm,
)


def validate_data_directory() -> None:
    """Raises a clear error if the ./data directory is missing or empty."""
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Data directory '{DATA_DIR}' not found. "
            "Please create it and add your documents."
        )
    files = os.listdir(DATA_DIR)
    if not files:
        raise FileNotFoundError(
            f"Data directory '{DATA_DIR}' is empty. "
            "Please add PDF, TXT, or DOCX files to it."
        )
    print(f"📁 Found {len(files)} file(s) in {DATA_DIR}: {files}")


def load_and_ingest_documents() -> list:
    """
    Loads documents from ./data and runs them through the enrichment pipeline.

    Uses batched processing with a sleep interval between batches to stay
    within Gemini free-tier rate limits (RPM constraints).

    Returns:
        list: Enriched nodes ready to be indexed.
    """
    validate_data_directory()

    print("📄 Loading documents...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"✅ Loaded {len(documents)} document(s).")

    # Build the transformation pipeline (no embed_model here — indexes handle that)
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
            QuestionsAnsweredExtractor(
                questions=METADATA_QUESTIONS_PER_NODE,
                llm=llm,
                num_workers=1,  # Sequential to respect rate limits
            ),
        ]
    )

    all_nodes = []
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"📦 Processing batch {batch_num}/{total_batches} "
              f"({len(batch)} document(s))...")

        try:
            batch_nodes = pipeline.run(documents=batch)
            all_nodes.extend(batch_nodes)
            print(f"   ✔ Batch {batch_num} produced {len(batch_nodes)} node(s).")
        except Exception as e:
            print(f"   ❌ Batch {batch_num} failed: {e}")
            raise

        # Sleep between batches to avoid hitting Gemini free-tier RPM limits.
        # Skip sleep after the last batch.
        if i + BATCH_SIZE < len(documents):
            print(f"⏳ Sleeping {BATCH_SLEEP_S}s to reset API quota...")
            time.sleep(BATCH_SLEEP_S)

    print(f"\n✅ Ingestion complete. Total nodes: {len(all_nodes)}")
    return all_nodes
