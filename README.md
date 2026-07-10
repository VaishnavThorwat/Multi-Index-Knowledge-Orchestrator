# 🧠 Multi-Index Knowledge Orchestrator

A **Retrieval-Augmented Generation (RAG)** system that lets you query your own documents using natural language. The orchestrator intelligently routes each query to the right retrieval strategy — **semantic search** for specific facts, or **tree summarization** for broad overviews — using a LangChain agent backed by Google Gemini.

---

## 🏗️ Architecture

<img width="2816" height="1536" alt="Gemini_Generated_Image_xvplgrxvplgrxvpl" src="https://github.com/user-attachments/assets/27853bcb-a0f5-4347-8239-f1fe52581c1a" />

**Dual-index design:** Both indexes are built from the *same* enriched nodes and share a single `StorageContext`, so they're persisted together and reload reliably from disk — no risk of one index going stale relative to the other.

**Agent-driven routing:** Rather than hardcoding which index to hit, a LangChain agent reads the user's query and decides — precise/factual questions go to the vector index, broad/thematic questions go to the summary index.

---

## ✨ Key Features

- **Two retrieval strategies, one interface** — semantic similarity search and tree-based summarization, auto-selected per query.
- **Shared storage context** — both indexes persist and load together from a single `./storage` directory, eliminating index drift.
- **Metadata-enriched nodes** — every chunk is enriched with `QuestionsAnsweredExtractor`-generated questions before indexing, improving retrieval relevance.
- **Free-tier–safe ingestion** — node-level batching with configurable sleep intervals keeps ingestion within Gemini API rate limits.
- **Safe query wrappers** — tool calls are wrapped with error handling so the agent degrades gracefully instead of crashing on network or query errors.

---

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/VaishnavThorwat/Multi-Index-Knowledge-Orchestrator.git
cd Multi-Index-Knowledge-Orchestrator
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
```
Open `.env` and add your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
Get a key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Add your documents
Place PDF, TXT, or DOCX files into the `./data` directory:
```bash
mkdir data
cp your_document.pdf data/
```

### 6. Run the orchestrator
```bash
python main.py
```

On first run, the app ingests and indexes your documents (this can take a few minutes depending on document count and API quota). Subsequent runs load instantly from `./storage`.

---

## 📁 Project Structure

```
Multi-Index-Knowledge-Orchestrator/
├── main.py          # Entry point — run this
├── config.py         # Model settings, API keys, constants
├── ingestion.py       # Document loading, splitting & metadata enrichment
├── indexes.py         # Dual-index build, persist, and load logic
├── agent.py            # LangChain tools & agent assembly
├── .env.example        # API key template (copy to .env)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

All tunable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-005` | Google embedding model |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model used for agent + LLM calls |
| `CHUNK_SIZE` | `512` | Token size per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `BATCH_SIZE` | `4` | Nodes processed per enrichment batch |
| `BATCH_SLEEP_S` | `65` | Sleep (seconds) between batches — keeps ingestion within free-tier rate limits |
| `METADATA_QUESTIONS_PER_NODE` | `3` | Metadata questions extracted per node |

> **Note on batching:** Batching happens at the *node* level, not the document level. The Gemini free-tier RPM limit is consumed by `QuestionsAnsweredExtractor` calls, which fire once per node — a handful of documents can expand into dozens of nodes, so document-level batching wouldn't actually protect the rate limit.

---

## 🧩 How Query Routing Works

The agent is given a system prompt with explicit tool-selection rules:

- **`semantic_search_tool`** → specific questions, facts, dates, names, technical details, or anything requiring a precise answer from the document.
- **`document_summary_tool`** → broad questions about overall content, main themes, general findings, or summaries of the entire document.

```python
queries = [
    "Summarize the key findings of the documents provided.",       # → summary tool
    "What are the main topics covered across all documents?",      # → summary tool
    "What is the policy number mentioned in section 3?",           # → semantic tool
    "When was the agreement signed?",                              # → semantic tool
]
```

Each tool call is wrapped in a safe-query handler that returns a clear fallback message instead of raising on empty results or network errors.

---

## 🛠️ Tech Stack

- **[LlamaIndex](https://docs.llamaindex.ai/)** — document indexing, node parsing, and query engines
- **[LangChain](https://python.langchain.com/)** — tool-calling agent orchestration
- **[Google Gemini](https://ai.google.dev/)** — LLM and embeddings (`gemini-2.0-flash`, `text-embedding-005`)

---

## ⚠️ Free Tier Notes

The Gemini free tier has rate limits (~15 RPM). The ingestion pipeline handles this automatically by enriching nodes in batches and sleeping `BATCH_SLEEP_S` seconds between batches. For larger document sets, consider upgrading to a paid tier or reducing `METADATA_QUESTIONS_PER_NODE` to `1`.

---

## 📄 License

MIT
