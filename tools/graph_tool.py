from langchain.tools import Tool
from neo4j import GraphDatabase
import os
import re # Import regex for basic entity extraction

# --- Neo4j Connection ---
driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
)

# --- Query Generation Logic ---
def generate_cypher_query(question: str) -> tuple[str | None, dict]:
    """
    Generates a Cypher query and parameters based on keywords in the question.
    Returns (cypher_query, params) or (None, {}) if no pattern matches.
    """
    question_lower = question.lower()
    params = {}

    # Pattern: "What functions does function X call?" / "What does X call?"
    match = re.search(r"(?:what|which) functions? does function (\w+)\s?call|what does (\w+)\s?call", question_lower)
    if match:
        func_name = match.group(1) or match.group(2)
        params['func_name'] = func_name
        cypher = """
        MATCH (caller:Function {name: $func_name})-[:CALLS]->(callee)
        RETURN caller.name AS caller, callee.name AS callee, callee.file_path AS callee_file
        LIMIT 15
        """
        return cypher, params

    # Pattern: "Who calls function X?" / "Which functions call X?"
    match = re.search(r"(?:who|which) functions? calls? function (\w+)", question_lower)
    if match:
        func_name = match.group(1)
        params['func_name'] = func_name
        cypher = """
        MATCH (caller)-[:CALLS]->(callee:Function {name: $func_name})
        RETURN caller.name AS caller, caller.file_path AS caller_file, callee.name AS callee
        LIMIT 15
        """
        return cypher, params

    # Pattern: "Where is variable X defined/declared?"
    match = re.search(r"where is (?:variable|var)\s+(\w+)\s+(?:defined|declared)", question_lower)
    if match:
        var_name = match.group(1)
        params['var_name'] = var_name
        cypher = """
        MATCH (scope)-[:DEFINES]->(v:Variable {name: $var_name})
        RETURN v.name AS variable, scope.scope_id AS defined_in_scope, v.defined_at AS line, v.definition_type AS type
        LIMIT 10
        """
        return cypher, params

    # Pattern: "Where is variable X used/read?"
    match = re.search(r"where is (?:variable|var)\s+(\w+)\s+(?:used|read)", question_lower)
    if match:
        var_name = match.group(1)
        params['var_name'] = var_name
        cypher = """
        MATCH (scope)-[r:READS]->(v:Variable {name: $var_name})
        RETURN v.name AS variable, scope.scope_id AS used_in_scope, r.lines AS lines
        LIMIT 10
        """
        return cypher, params

    # Pattern: "What does function X return?"
    match = re.search(r"what does function (\w+)\s?return", question_lower)
    if match:
        func_name = match.group(1)
        params['func_name'] = func_name
        cypher = """
        MATCH (fn:Function {name: $func_name})-[r:RETURNS]->()
        WHERE r.lines IS NOT NULL
        RETURN fn.name AS function, r.lines AS return_lines, r.value_repr AS returned_value_representation
        LIMIT 5
        """
        return cypher, params

    # Pattern: "What modules does file X import?" (Requires full path or unique filename)
    # This is harder with regex, let's match on 'import' and assume the last word might be a file path part
    if "import" in question_lower and ("file" in question_lower or "module" in question_lower):
         # Very basic: assumes file path might be mentioned. Needs improvement.
         # Example: "What does main.py import?"
         parts = question.split()
         potential_file = parts[-1].replace('?','') # Crude extraction
         params['file_path_part'] = potential_file
         cypher = """
         MATCH (f:File)-[rel:IMPORTS|IMPORTS_FROM]->(m:Module)
         WHERE f.path ENDS WITH $file_path_part // Use ENDS WITH for partial matching
         RETURN DISTINCT f.path as file, type(rel) as import_type, m.name as module, m.type as module_type, rel.alias as alias
         LIMIT 20
         """
         return cypher, params

    # Pattern: "Which classes inherit from class X?"
    match = re.search(r"(?:which|what) classes? inherits? from class (\w+)", question_lower)
    if match:
        base_class_name = match.group(1)
        params['base_class_name'] = base_class_name
        cypher = """
        MATCH (c:Class)-[:INHERITS_FROM]->(base:Class {name: $base_class_name})
        RETURN c.name AS subclass, c.file_path AS file
        LIMIT 10
        """
        return cypher, params

    # Fallback / Default (optional) - maybe simple function search
    # match = re.search(r"find function (\w+)", question_lower)
    # if match:
    #     func_name = match.group(1)
    #     params['func_name'] = func_name
    #     cypher = """
    #     MATCH (f:Function {name: $func_name})
    #     RETURN f.name as function, f.file_path as file, f.line as line
    #     LIMIT 5
    #     """
    #     return cypher, params

    return None, {} # No query generated

# --- Tool Execution Logic ---
def execute_graph_query(question: str) -> str:
    """
    Generates a Cypher query based on the question, executes it, and formats the result.
    """
    cypher, params = generate_cypher_query(question)

    if not cypher:
        return "Sorry, I couldn't understand that question structure for the graph database. Try asking about function calls, variable definitions/usage, imports, or inheritance."

    print(f"🧠 Generated Cypher:\n{cypher}\nParams: {params}") # Log the generated query

    try:
        with driver.session() as session:
            results = session.run(cypher, params)
            records = list(results) # Consume results

            if not records:
                return "No results found in the graph database for that query."

            # Format results into a readable string
            formatted_lines = []
            headers = results.keys()
            formatted_lines.append(" | ".join(headers))
            formatted_lines.append("-" * (sum(len(h) for h in headers) + len(headers)*3 -1)) # Separator line

            for record in records:
                line_parts = []
                for key in headers:
                    value = record[key]
                    # Truncate long strings/lists for readability
                    if isinstance(value, list):
                        value_str = repr(value[:3]) + ('...' if len(value) > 3 else '')
                    elif isinstance(value, str) and len(value) > 50:
                         value_str = value[:47] + '...'
                    else:
                         value_str = str(value)
                    line_parts.append(value_str)
                formatted_lines.append(" | ".join(line_parts))

            return "\n".join(formatted_lines)

    except Exception as e:
        print(f"❌ Error executing Cypher: {e}")
        return f"An error occurred while querying the graph database: {e}"

# --- LangChain Tool Definition ---
graph_tool = Tool(
    name="CodeGraphExplorer",
    func=execute_graph_query, # Use the new execution function
    description="Use this tool to explore the codebase structure and relationships. Ask questions about: function calls (e.g., 'what does function X call?', 'who calls function Y?'), variable definitions ('where is variable Z defined?'), variable usage ('where is variable Z used?'), file imports ('what does file A import?'), or class inheritance ('which classes inherit from class B?'). Provide specific names for functions, variables, classes, or files."
)

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Test cases
    questions = [
        "What functions does main call?",
        "Who calls ingest_to_upstash?",
        "Where is variable api_key defined?",
        "Where is variable docs used?",
        "What does load_and_split_files return?",
        "What modules does file main.py import?",
        "Which classes inherit from NodeVisitor?", # Will likely return nothing unless you ingested langchain code
        "find function scan_project", # Test fallback if uncommented
        "Tell me about the weather" # Test no match
    ]
    for q in questions:
        print(f"\n--- Question: {q} ---")
        result = execute_graph_query(q)
        print(result)

    driver.close()
