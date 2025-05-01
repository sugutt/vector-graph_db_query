import os
from langchain.tools import Tool
from langchain_together import ChatTogether # To synthesize results

# Import the core functions from existing tools (assuming they can be imported)
# Adjust imports based on your actual file structure and function names
from tools.vector_tool import vector_search # Assuming vector_search(query) returns formatted string
from tools.graph_tool import execute_graph_query # Assuming execute_graph_query(query) returns formatted string

# Initialize LLM for synthesis
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",    # Or your preferred model
    temperature=0.1, # Low temp for synthesis
    together_api_key=os.environ["TOGETHER_API_KEY"]
)

def hybrid_code_search(query: str) -> str:
    """
    Performs both semantic (vector) and structural (graph) searches
    based on the user query and synthesizes the results using an LLM.
    """
    print(f"⚡ Performing Hybrid Search for: '{query}'")

    # 1. Perform Vector Search
    print("  - Performing vector search...")
    try:
        vector_results = vector_search(query)
        print(f"  Vector Results:\n{vector_results[:500]}...") # Log snippet
    except Exception as e:
        print(f"  ⚠️ Error during vector search: {e}")
        vector_results = "Vector search failed."

    # 2. Perform Graph Search
    print("  - Performing graph search...")
    try:
        # execute_graph_query handles query generation internally
        graph_results = execute_graph_query(query)
        print(f"  Graph Results:\n{graph_results[:500]}...") # Log snippet
    except Exception as e:
        print(f"  ⚠️ Error during graph search: {e}")
        graph_results = "Graph search failed or query structure not understood."

    # 3. Synthesize Results using LLM
    print("  - Synthesizing results...")
    synthesis_prompt = f"""
Original User Query: "{query}"

Based on the following information retrieved from the codebase, provide a comprehensive answer to the user's query.

Semantic Search Results (relevant code snippets):
---
{vector_results}
---

Structural Graph Search Results (code relationships, definitions, calls):
---
{graph_results}
---

Synthesized Answer:
"""
    try:
        response = llm.invoke(synthesis_prompt)
        synthesized_answer = response.content # Adjust based on actual response object structure
        print(f"  Synthesized Answer:\n{synthesized_answer[:500]}...")
        return synthesized_answer
    except Exception as e:
        print(f"  ❌ Error during LLM synthesis: {e}")
        return f"Failed to synthesize results. Vector search found:\n{vector_results}\nGraph search found:\n{graph_results}"


# --- LangChain Tool Definition ---
hybrid_tool = Tool(
    name="HybridCodeSearch",
    func=hybrid_code_search,
    description="Use this tool to answer questions about the codebase. It combines semantic search for relevant code snippets and graph search for code structure (functions, classes, calls, variables, imports). Provide a natural language query describing what you want to know about the code."
)

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    test_query = "Show me the function that loads files and tell me where the 'docs' variable is used"
    # test_query = "How is the Together API key used?"
    result = hybrid_code_search(test_query)
    print("\n--- Final Result ---")
    print(result)