"""
ingestion.py
------------
Handles document loading from ./data and runs the enrichment pipeline.

Pipeline steps:
  1. SentenceSplitter            - chunks documents into overlapping text
                                    nodes. Makes no LLM calls; runs once on
                                    the full document set.
  2. QuestionsAnsweredExtractor  - enriches each node with metadata
                                    questions. Makes ONE LLM call per node -
                                    this is the step that must be rate-limited.

NOTE: Embedding is intentionally NOT part of this pipeline.
      The VectorStoreIndex handles embedding automatically when
      nodes are passed to it, avoiding double-embedding.

RATE LIMITING (fixed):
      Batching previously happened at the *document* level, but the Gemini
      free-tier RPM limit is consumed by QuestionsAnsweredExtractor calls,
      which happen once per *node*. A handful of documents can easily
      expand into dozens of nodes, so document-level batching did not
      actually protect the rate limit - a single batch of "4 documents"
      could fire 20+ LLM calls before the sleep ever kicked in.

      Batching now happens at the node level instead: documents are split
      into nodes up front (cheap, zero API calls), then nodes are enriched
      in batches of BATCH_SIZE with a BATCH_SLEEP_S pause between batches.
      BATCH_SIZE therefore now means "nodes per extraction batch," not
      "documents per batch" - update the comment in config.py to match.
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
    print(f"Found {len(files)} file(s) in {DATA_DIR}: {files}")


def _split_documents_into_nodes(documents: list) -> list:
    """
    Splits documents into nodes. This step makes no LLM calls, so it's
    safe to run on the entire document set in one go - no rate-limit
    concerns here.
    """
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(nodes)} node(s).")
    return nodes


def _enrich_nodes_in_batches(nodes: list) -> list:
    """
    Runs QuestionsAnsweredExtractor over nodes in batches, sleeping
    between batches to stay within Gemini free-tier RPM limits.

    This is where the actual rate-limit fix lives: batching happens at
    the node level because that's where the LLM calls occur (one per
    node), not at the document level.
    """
    pipeline = IngestionPipeline(
        transformations=[
            QuestionsAnsweredExtractor(
                questions=METADATA_QUESTIONS_PER_NODE,
                llm=llm,
                num_workers=1,  # Sequential to respect rate limits
            ),
        ]
    )

    enriched_nodes = []
    total_batches = (len(nodes) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Enriching batch {batch_num}/{total_batches} "
              f"({len(batch)} node(s))...")

        try:
            enriched_batch = pipeline.run(nodes=batch)
            enriched_nodes.extend(enriched_batch)
            print(f"   Batch {batch_num} enriched.")
        except Exception as e:
            print(f"   Batch {batch_num} failed: {e}")
            print(f"   {len(enriched_nodes)} node(s) enriched before the "
                  "failure are kept in memory, but this function still "
                  "raises - the caller does not yet persist partial "
                  "progress. (That's a separate fix if you want it.)")
            raise

        # Sleep between batches to avoid hitting Gemini free-tier RPM limits.
        # Skip sleep after the last batch.
        if i + BATCH_SIZE < len(nodes):
            print(f"Sleeping {BATCH_SLEEP_S}s to reset API quota...")
            time.sleep(BATCH_SLEEP_S)

    return enriched_nodes


def load_and_ingest_documents() -> list:
    """
    Loads documents from ./data and runs them through the enrichment pipeline.

    Splitting happens once, up front (no API calls). Enrichment then runs
    in node-level batches with a sleep interval between batches to stay
    within Gemini free-tier rate limits (RPM constraints).

    Returns:
        list: Enriched nodes ready to be indexed.
    """
    validate_data_directory()

    print("Loading documents...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    print(f"Loaded {len(documents)} document(s).")

    nodes = _split_documents_into_nodes(documents)
    enriched_nodes = _enrich_nodes_in_batches(nodes)

    print(f"\nIngestion complete. Total nodes: {len(enriched_nodes)}")
    return enriched_nodes
