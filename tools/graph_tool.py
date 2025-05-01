from langchain.tools import Tool
from neo4j import GraphDatabase
import os

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
)

def graph_query(question: str) -> str:
    """
    Translate a limited question to a Cypher query. (Simplified version)
    """
    cypher = """
    MATCH (f:Function)-[:CALLS]->(c:Function)
    WHERE f.name CONTAINS $q OR c.name CONTAINS $q
    RETURN f.name AS caller, c.name AS callee LIMIT 10
    """
    with driver.session() as session:
        results = session.run(cypher, q=question)
        lines = [f"{r['caller']} → {r['callee']}" for r in results]
        return "\n".join(lines) or "No relationships found."

graph_tool = Tool(
    name="CodeGraphExplorer",
    func=graph_query,
    description="Use this to explore function relationships in the codebase."
)
