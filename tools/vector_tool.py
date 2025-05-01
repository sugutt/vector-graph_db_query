import os
from langchain.tools import Tool
from langchain_community.vectorstores.upstash import UpstashVectorStore
from langchain_together import TogetherEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-32k-retrieval", 
    api_key=os.environ["TOGETHER_API_KEY"]
)

# Connect to the existing Upstash index by instantiating the class
# It will use UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN from env vars
vectorstore = UpstashVectorStore(embedding=embeddings)

# Create the retriever
retriever = vectorstore.as_retriever()

# Define the tool function
def vector_search(query: str) -> str:
    """Searches the vector database for relevant code snippets based on the query."""
    docs = retriever.get_relevant_documents(query)
    # Format the results (adjust as needed)
    results = "\n---\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n```\n{doc.page_content}\n```" for doc in docs])
    return f"Found relevant code snippets:\n{results}"

# Create the LangChain tool
vector_tool = Tool(
    name="VectorCodeSearch",
    func=vector_search,
    description="Use this tool to search for code snippets based on a natural language query. Input should be the search query."
)
