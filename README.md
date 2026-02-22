# 🔍 RAG Research Agent

A **Retrieval-Augmented Generation (RAG)** system that lets you query your own documents using natural language. Built with LlamaIndex, LangChain, and Google Gemini.

The agent intelligently selects between **semantic search** (for specific facts) and **tree summarization** (for overviews) based on the nature of your question.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
LangChain Agent (Gemini 2.0 Flash)
    ├── semantic_search_tool  ──► VectorStoreIndex  (specific facts & details)
    └── document_summary_tool ──► SummaryIndex      (overviews & themes)
                                        │
                              Enriched Nodes (via IngestionPipeline)
                                        │
                              SentenceSplitter + QuestionsAnsweredExtractor
                                        │
                                  Your Documents (./data)
```

**Dual-index design:** Both indexes are built from the same enriched nodes and share a single StorageContext, ensuring both are persisted and loaded reliably.

---

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-research-agent.git
cd rag-research-agent
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
# Now open .env and add your Gemini API key
```
Get your key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Add your documents
Place PDF, TXT, or DOCX files into the `./data` directory:
```bash
mkdir data
cp your_document.pdf data/
```

### 6. Run the agent
```bash
python main.py
```

On first run, the app will index your documents (this may take a few minutes depending on document count and API quota). Subsequent runs load instantly from `./storage`.

---

## 📁 Project Structure

```
rag-research-agent/
├── main.py          # Entry point — run this
├── config.py        # Model settings, API keys, constants
├── ingestion.py     # Document loading & enrichment pipeline
├── indexes.py       # Dual-index build, persist, and load logic
├── agent.py         # LangChain tools & agent assembly
├── .env.example     # API key template (copy to .env)
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
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model for LLM calls |
| `CHUNK_SIZE` | `512` | Token size per text chunk |
| `BATCH_SIZE` | `4` | Documents processed per batch |
| `BATCH_SLEEP_S` | `65` | Sleep between batches (free-tier safe) |
| `METADATA_QUESTIONS_PER_NODE` | `3` | Metadata questions extracted per chunk |

---

## 🛠️ Tech Stack

- **[LlamaIndex](https://docs.llamaindex.ai/)** — Document indexing, node parsing, and query engines
- **[LangChain](https://python.langchain.com/)** — Agent orchestration and tool management
- **[Google Gemini](https://ai.google.dev/)** — LLM and embeddings (`gemini-2.0-flash`, `text-embedding-005`)

---

## ⚠️ Free Tier Notes

The Gemini free tier has rate limits (~15 RPM). The ingestion pipeline handles this automatically by processing documents in batches and sleeping 65 seconds between batches. For larger document sets, consider upgrading to a paid tier or reducing `METADATA_QUESTIONS_PER_NODE` to 1.

---

## 📄 License

MIT
