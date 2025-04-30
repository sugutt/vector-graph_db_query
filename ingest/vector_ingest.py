import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.upstash import UpstashVectorStore
from langchain_together import TogetherEmbeddings
from dotenv import load_dotenv
import sys

load_dotenv()
from tqdm import tqdm

from ingest.scan import scan_project

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
def load_and_split_files(file_paths):
    docs = []
    for path in tqdm(file_paths, desc="📄 Splitting files"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Change 'metadata' to 'metadatas'
            chunks = text_splitter.create_documents([content], metadatas=[{"source": str(path)}]) # Also ensure path is a string
            docs.extend(chunks)
        except Exception as e:
            print(f"⚠️ Skipped {path}: {e}")
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
