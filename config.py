"""
config.py
---------
Central configuration for the RAG Research Agent.
All environment variables and shared constants are defined here.
"""

import os
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# --- API Key ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please add it to your .env file.\n"
        "See .env.example for the required format."
    )

# --- Model Configuration ---
# Updated to latest models as of 2026
EMBEDDING_MODEL = "text-embedding-005"          # Latest stable embedding model
LLM_MODEL      = "gemini-2.0-flash"             # Latest fast model (replaces 1.5-flash)

# --- Storage & Data Paths ---
PERSIST_DIR   = "./storage"
DATA_DIR      = "./data"

# --- Index IDs (used to retrieve specific indexes from shared storage) ---
VECTOR_INDEX_ID  = "vector_index_001"
SUMMARY_INDEX_ID = "summary_index_001"

# --- Ingestion Settings ---
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
BATCH_SIZE    = 4    # Documents per batch (tune based on your API quota)
BATCH_SLEEP_S = 65   # Seconds to sleep between batches (free-tier safe)
METADATA_QUESTIONS_PER_NODE = 3

# --- Initialize Models ---
embed_model = GoogleGenAIEmbedding(
    model_name=EMBEDDING_MODEL,
    api_key=API_KEY
)

llm = GoogleGenAI(
    model=LLM_MODEL,
    api_key=API_KEY
)

agent_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=API_KEY
)

# --- Apply Global LlamaIndex Settings ---
Settings.embed_model = embed_model
Settings.llm = llm
