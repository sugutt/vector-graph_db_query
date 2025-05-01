import argparse
from ingest.scan import scan_project
from config import DEFAULT_PROJECT_PATH
from ingest.vector_ingest import load_and_split_files, ingest_to_upstash
from ingest.graph_builder import CodeGraphBuilder
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Scan and process a codebase.")
    parser.add_argument("--path", type=str, default=str(DEFAULT_PROJECT_PATH), help="Path to codebase")
    args = parser.parse_args()
    project_path = Path(args.path).resolve()

    print(f"🔍 Scanning codebase at {project_path}...\n")
    files = scan_project(project_path)

    for f in files:
        print("📄", f)

    print(f"\n✅ Found {len(files)} code files.")

    # --- Vector Ingestion ---
    print("\n⚡ Starting Vector DB Ingestion...")
    docs = load_and_split_files(files)
    if docs:
        ingest_to_upstash(docs)
    else:
        print("⚠️ No documents generated for vector ingestion.")

    # --- Graph Ingestion ---
    print("\n🕸️ Starting Graph DB Ingestion...")
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("❌ Missing Neo4j credentials in environment variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD). Skipping graph ingestion.")
        return

    graph = CodeGraphBuilder(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )

    try:
        graph.clear_graph()
        graph.build_graph_from_files(files, project_path)
    except Exception as e:
        print(f"❌ An error occurred during graph building: {e}")
    finally:
        graph.close()
        print("🚪 Closed graph database connection.")


if __name__ == "__main__":
    main()
