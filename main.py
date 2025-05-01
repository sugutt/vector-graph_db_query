import argparse
from ingest.scan import scan_project
from config import DEFAULT_PROJECT_PATH
from ingest.vector_ingest import load_and_split_files, ingest_to_upstash
from ingest.graph_builder import CodeGraphBuilder
import os

def main():
    parser = argparse.ArgumentParser(description="Scan and process a codebase.")
    parser.add_argument("--path", type=str, default=DEFAULT_PROJECT_PATH, help="Path to codebase")
    args = parser.parse_args()

    print(f"🔍 Scanning codebase at {args.path}...\n")
    files = scan_project(args.path)

    for f in files:
        print("📄", f)

    print(f"\n✅ Found {len(files)} code files.")

    # After scanning files...
    docs = load_and_split_files(files)
    ingest_to_upstash(docs)

    graph = CodeGraphBuilder(
    uri=os.environ["NEO4J_URI"],
    user=os.environ["NEO4J_USER"],
    password=os.environ["NEO4J_PASSWORD"],
    )

    graph.clear_graph()
    graph.build_graph_from_files(files)
    graph.close()


if __name__ == "__main__":
    main()
