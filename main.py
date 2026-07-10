"""
main.py
-------
Entry point for the RAG Research Agent.

Run with:
    python main.py

Or import `run_query()` to integrate into another application.
"""

from indexes import get_indexes
from agent import build_agent


def run_query(query: str, verbose: bool = True) -> str:
    """
    Loads indexes and runs a single query through the agent.

    Args:
        query:   The user's natural language question.
        verbose: Whether to print agent reasoning steps.

    Returns:
        The agent's final response string.
    """
    vector_index, summary_index = get_indexes()
    agent = build_agent(vector_index, summary_index, verbose=verbose)

    print(f"\nQuery: {query}")
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    response = result["messages"][-1].content
    print(f"\nAgent Response:\n{response}")
    return response


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Add your queries here. A few examples to demonstrate both tools:
    # -----------------------------------------------------------------------
    queries = [
        "Summarize the key findings of the documents provided.",       # uses summary tool
        "What are the main topics covered across all documents?",      # uses summary tool
        # "What is the policy number mentioned in section 3?",         # uses semantic tool
        # "When was the agreement signed?",                            # uses semantic tool
    ]

    for q in queries:
        run_query(q)
        print("\n" + "=" * 60 + "\n")
