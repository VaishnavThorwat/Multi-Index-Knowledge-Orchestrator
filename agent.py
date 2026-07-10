"""
agent.py
--------
Defines the LangChain agent with two RAG-backed tools:

  1. semantic_search_tool  - Vector similarity search for specific facts.
  2. document_summary_tool - Tree summarization for overviews/themes.

The agent (powered by Gemini) intelligently decides which tool to use
based on the nature of the user's query.
"""

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langgraph.graph.state import CompiledStateGraph

from llama_index.core import VectorStoreIndex, SummaryIndex

from config import agent_llm

# --- System Prompt ---
SYSTEM_PROMPT = (
    "You are a professional Research Assistant with access to a document knowledge base.\n\n"
    "Tool selection rules:\n"
    "- Use 'semantic_search_tool' for specific questions, facts, dates, names, "
    "technical details, or anything requiring a precise answer from the document.\n"
    "- Use 'document_summary_tool' for broad questions about the overall content, "
    "main themes, general findings, or summaries of the entire document.\n\n"
    "Always base your answers strictly on the documents. "
    "If the answer cannot be found, say so clearly."
)


def _safe_query(engine, query: str, tool_name: str) -> str:
    """
    Wraps a query engine call with error handling so agent doesn't crash
    on network issues or empty results.
    """
    try:
        result = engine.query(query)
        response = result.response
        if not response or response.strip().lower() in ("none", ""):
            return "No relevant information found in the documents for this query."
        return response
    except Exception as e:
        return f"[{tool_name} error] Could not retrieve a result: {str(e)}"


def build_agent(
    vector_index: VectorStoreIndex,
    summary_index: SummaryIndex,
    verbose: bool = True,
) -> CompiledStateGraph:
    """
    Builds and returns a LangChain AgentExecutor with dual RAG tools.

    Args:
        vector_index:  The VectorStoreIndex for semantic search.
        summary_index: The SummaryIndex for document summarization.
        verbose:       Whether to print agent reasoning steps.

    Returns:
        A configured LangChain agent graph ready to invoke.
    """
    # Query engines
    vector_engine  = vector_index.as_query_engine(similarity_top_k=3)
    summary_engine = summary_index.as_query_engine(response_mode="tree_summarize")

    # Tools with safe wrappers
    tools = [
        Tool(
            name="semantic_search_tool",
            func=lambda q: _safe_query(vector_engine, q, "semantic_search_tool"),
            description=(
                "Search for specific facts, technical details, dates, names, "
                "policy numbers, or any precise information within the documents."
            ),
        ),
        Tool(
            name="document_summary_tool",
            func=lambda q: _safe_query(summary_engine, q, "document_summary_tool"),
            description=(
                "Use this when the user asks for an overview, summary, main themes, "
                "key findings, or general understanding of the entire document."
            ),
        ),
    ]

    # Assemble agent
    return create_agent(
        model=agent_llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        debug=verbose,
    )
