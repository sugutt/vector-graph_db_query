import ast
import os
from neo4j import GraphDatabase
from tqdm import tqdm

class CodeGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def build_graph_from_files(self, file_paths):
        for path in tqdm(file_paths, desc="🧠 Building Graph"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content)
                self._parse_ast(tree, path)
            except Exception as e:
                print(f"⚠️ Failed to parse {path}: {e}")

    def _parse_ast(self, tree, file_path):
        with self.driver.session() as session:
            session.run("MERGE (f:File {path: $path})", {"path": file_path})
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    session.run("""
                        MERGE (func:Function {name: $name, file: $file})
                        MERGE (f:File {path: $file})-[:CONTAINS]->(func)
                    """, {"name": node.name, "file": file_path})
                    for call in [n for n in ast.walk(node) if isinstance(n, ast.Call)]:
                        if isinstance(call.func, ast.Name):
                            called = call.func.id
                            session.run("""
                                MATCH (a:Function {name: $caller, file: $file})
                                MERGE (b:Function {name: $callee})
                                MERGE (a)-[:CALLS]->(b)
                            """, {
                                "caller": node.name,
                                "callee": called,
                                "file": file_path
                            })
