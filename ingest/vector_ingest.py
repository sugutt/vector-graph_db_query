import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.upstash import UpstashVectorStore
from langchain_together import TogetherEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path  # Ensure Path is imported
from multiprocessing import Pool, cpu_count  # <-- Import multiprocessing
import functools  # <-- Import functools for partial

load_dotenv()

# --- Global splitter (used by worker function) ---
# Ensure chunk_size and chunk_overlap are defined appropriately
TEXT_SPLITTER_CHUNK_SIZE = 500
TEXT_SPLITTER_CHUNK_OVERLAP = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
    chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP
)
# -------------------------------------------------

def _process_file_for_splitting(file_path: Path) -> list:
    """Worker function to read and split a single file."""
    try:
        # Ensure path is a Path object
        path_obj = Path(file_path) if not isinstance(file_path, Path) else file_path
        with open(path_obj, "r", encoding="utf-8") as f:
            content = f.read()
        # Use the globally defined text_splitter
        chunks = text_splitter.create_documents([content], metadatas=[{"source": str(path_obj)}])
        return chunks
    except Exception as e:
        print(f"⚠️ Worker failed to split {file_path}: {e}")
        return []

def load_and_split_files(file_paths: list[Path]) -> list:
    """Loads and splits files in parallel using multiprocessing."""
    docs = []
    num_files = len(file_paths)
    if num_files == 0:
        return docs

    # Determine number of processes (leave one CPU free)
    num_workers = max(1, cpu_count() - 1)
    print(f"📄 Splitting {num_files} files using {num_workers} worker processes...")

    # Use Pool.imap_unordered for progress bar and better memory usage
    with Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(_process_file_for_splitting, file_paths)
        # Wrap with tqdm for progress bar
        for result_chunks in tqdm(results_iterator, total=num_files, desc="📄 Splitting files"):
            docs.extend(result_chunks)

    print(f"📄 Generated {len(docs)} document chunks.")
    return docs

def ingest_to_upstash(docs):
    # Load the API key from the environment variable
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY is not set in the environment variables.")

    # Initialize the embeddings with the API key
    embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-32k-retrieval", api_key=api_key)

    # Pass the embeddings to the vector store
    vectorstore = UpstashVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,  # Correct argument name is 'embedding'
    )
    print("✅ Vector DB ingestion complete.")
    return vectorstore
