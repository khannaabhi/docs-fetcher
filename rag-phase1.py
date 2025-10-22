import os
import asyncio
from openai import OpenAI
import lancedb
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer # New import for local embeddings
import glob # For finding files in directories
import shutil # For removing directories

# --- 0. Configuration and Setup ---
# Replace with your actual OpenAI API Key.
# It's recommended to set this as an environment variable (e.g., OPENAI_API_KEY).
# For this example, we'll use a placeholder.
# You can set it like: os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
openai_api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

if openai_api_key == "YOUR_OPENAI_API_KEY_HERE":
    print("WARNING: Please set your OPENAI_API_KEY environment variable or replace 'YOUR_OPENAI_API_KEY_HERE' with your actual key.")
    print("OpenAI embeddings might not work without a valid key.")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Define the path for your LanceDB database
LANCEDB_PATH = "lancedb_rag_data"
TABLE_NAME = "documents"

# --- IMPORTANT: Clean up previous LanceDB data for fresh start ---
if os.path.exists(LANCEDB_PATH):
    print(f"Removing existing LanceDB directory: {LANCEDB_PATH}")
    shutil.rmtree(LANCEDB_PATH)
    print("Existing LanceDB directory removed.")

print(f"LanceDB will be stored at: {LANCEDB_PATH}")

# --- Local Embedding Model Configuration ---
# Set to True to use local embeddings, False to use OpenAI embeddings
USE_LOCAL_EMBEDDINGS = True
LOCAL_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A good default small model

# Initialize local embedding model (if USE_LOCAL_EMBEDDINGS is True)
local_embedding_model = None
if USE_LOCAL_EMBEDDINGS:
    try:
        print(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL_NAME}...")
        local_embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL_NAME)
        print("Local embedding model loaded successfully.")
    except Exception as e:
        print(f"Error loading local embedding model: {e}")
        print("Falling back to OpenAI embeddings if available.")
        USE_LOCAL_EMBEDDINGS = False # Disable local embeddings if loading fails


# --- Helper Functions for Embedding ---
async def get_openai_embeddings_async(texts):
    """
    Asynchronously gets embeddings for a list of texts using OpenAI's API.
    """
    try:
        response = await client.embeddings.create(
            input=texts,
            model="text-embedding-3-small" # Or "text-embedding-3-large" for higher quality
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error generating OpenAI embeddings: {e}")
        return [None] * len(texts)

async def get_local_embeddings_async(texts, model):
    """
    Asynchronously gets embeddings for a list of texts using a local SentenceTransformer model.
    """
    if model is None:
        print("Local embedding model is not loaded. Cannot generate local embeddings.")
        return [None] * len(texts)
    try:
        embeddings = model.encode(texts, convert_to_list=True)
        return embeddings
    except Exception as e:
        print(f"Error generating local embeddings: {e}")
        return [None] * len(texts)

async def get_embeddings(texts, use_local, local_model):
    """
    Chooses between local and OpenAI embeddings based on the flag.
    """
    if use_local:
        print(f"Using local embedding model: {LOCAL_EMBEDDING_MODEL_NAME}")
        return await get_local_embeddings_async(texts, local_model)
    else:
        print("Using OpenAI embedding model.")
        return await get_openai_embeddings_async(texts)


# --- 1. Data Gatherer (Reading from Nested Directory) ---
# Create a dummy nested directory and README files for demonstration
DATA_DIR = "docs"
os.makedirs(os.path.join(DATA_DIR, "project_a"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "project_b", "sub_project"), exist_ok=True)

with open(os.path.join(DATA_DIR, "project_a", "README.md"), "w") as f:
    f.write("""# Project A Overview
This is the main README for Project A. It focuses on developing a new machine learning algorithm for anomaly detection in time series data.
The algorithm uses a combination of recurrent neural networks and statistical methods.
Key features include real-time processing and high accuracy.
""")

with open(os.path.join(DATA_DIR, "project_a", "INSTALL.md"), "w") as f:
    f.write("""# Installation Guide for Project A
To install Project A, ensure you have Python 3.9+ and pip.
1. Clone the repository: `git clone https://github.com/yourorg/project_a.git`
2. Navigate to the directory: `cd project_a`
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest`
""")

with open(os.path.join(DATA_DIR, "project_b", "README.md"), "w") as f:
    f.write("""# Project B: Data Visualization Tool
Project B is a web-based tool for visualizing large datasets. It supports various chart types including bar charts, line graphs, and scatter plots.
It's built using React for the frontend and FastAPI for the backend.
Users can upload CSV files and interactively explore their data.
""")

with open(os.path.join(DATA_DIR, "project_b", "sub_project", "README.md"), "w") as f:
    f.write("""# Sub-Project within Project B: Real-time Dashboard
This sub-project focuses on building a real-time dashboard component for Project B.
It uses WebSockets to push live data updates to the client.
Technologies involved are D3.js for interactive elements and Redis for caching.
""")

print(f"\n--- Reading Documents from '{DATA_DIR}' ---")

def read_markdown_files(directory):
    """Reads all .md files from a directory and its subdirectories."""
    file_contents = []
    file_paths = glob.glob(os.path.join(directory, "**", "*.md"), recursive=True)
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Store content along with its original file path for better context
                file_contents.append({"text": content, "source_path": file_path})
            print(f"Read: {file_path}")
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
    return file_contents

raw_documents = read_markdown_files(DATA_DIR)
print(f"Found {len(raw_documents)} Markdown files.")

# --- 2. RAG Phase: Semantic Chunking, Embedding, and Storage ---
print("\n--- Step 1-3: Semantic Chunking, Embedding, and Storing in LanceDB ---")

# Initialize the SemanticChunker
chunker = SemanticChunker(
    min_chunk_size=50,  # Minimum tokens per chunk
    chunk_size=200, # Maximum tokens per chunk
)

all_documents_to_store = []

for doc_idx, raw_doc in enumerate(raw_documents):
    doc_text = raw_doc["text"]
    doc_source_path = raw_doc["source_path"]

    print(f"\nProcessing document from: {doc_source_path}")
    chunks = chunker.chunk(doc_text)
    print(f"  Generated {len(chunks)} chunks for this document.")

    # Extract just the text from the chunks for embedding
    chunk_texts = [chunk.text for chunk in chunks]

    # Get embeddings for the current document's chunks
    embeddings = asyncio.run(get_embeddings(chunk_texts, USE_LOCAL_EMBEDDINGS, local_embedding_model))

    for i, chunk in enumerate(chunks):
        if embeddings[i] is not None: # Only add if embedding was successful
            # Ensure 'id' is always present and unique across all documents
            # Combine original document path with chunk index for a unique ID
            doc_id = f"{doc_source_path}_{i}"
            all_documents_to_store.append({
                "id": doc_id,
                "text": chunk.text,
                "source_path": doc_source_path, # Store source path as metadata
                "vector": embeddings[i]
            })
        else:
            print(f"  Skipping chunk {i} from {doc_source_path} due to embedding failure.")

# Connect to LanceDB and store all processed documents
db = lancedb.connect(LANCEDB_PATH)

try:
    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        print(f"\nTable '{TABLE_NAME}' already exists. Appending new data.")
        table.add(all_documents_to_store)
    else:
        table = db.create_table(TABLE_NAME, data=all_documents_to_store)
        print(f"\nTable '{TABLE_NAME}' created and data added.")

    print(f"Total documents in LanceDB table '{TABLE_NAME}': {table.count_rows()}")

except Exception as e:
    print(f"Error storing data in LanceDB: {e}")

# --- 5. RAG Phase: Retrieval ---
print("\n--- Step 4: Retrieval from LanceDB ---")

async def retrieve_chunks(query_text: str, k: int = 3):
    """
    Retrieves the top-k most relevant chunks from LanceDB for a given query.
    """
    print(f"\nRetrieving for query: '{query_text}'")
    # 1. Embed the query
    # IMPORTANT: Use 'await' here, not 'asyncio.run()', as this function is already
    # being called within an active asyncio event loop.
    query_embeddings_list = await get_embeddings([query_text], USE_LOCAL_EMBEDDINGS, local_embedding_model)

    # Robustly check if an embedding was successfully generated and convert to list if it's a numpy array
    query_vector = None
    if query_embeddings_list.any() and len(query_embeddings_list) > 0 and query_embeddings_list[0] is not None:
        # Check if the embedding is a numpy array and convert it to a list
        if hasattr(query_embeddings_list[0], 'tolist'):
            query_vector = query_embeddings_list[0].tolist()
        else:
            query_vector = query_embeddings_list[0]
    else:
        print("Failed to generate embedding for the query. Query vector is None or empty.")
        return []

    # 2. Perform similarity search in LanceDB
    try:
        table = db.open_table(TABLE_NAME)
        # Use the query vector to search for similar documents
        results = table.search(query_vector).limit(k).to_list()
        print(f"Found {len(results)} relevant chunks.")
        return results
    except Exception as e:
        print(f"Error during LanceDB retrieval: {e}")
        return []

# --- Demonstration of Retrieval ---
sample_query_1 = "How to install Project A?"
sample_query_2 = "What is Project B about and its technologies?"
sample_query_3 = "Tell me about the real-time dashboard."

print("\n--- Running Sample Queries ---")

# These top-level calls correctly use asyncio.run() to start the event loop
retrieved_chunks_1 = asyncio.run(retrieve_chunks(sample_query_1))
print(f"\nResults for '{sample_query_1}':")
for i, chunk in enumerate(retrieved_chunks_1):
    print(f"  Chunk {i+1} (ID: {chunk['id']}, Source: {chunk['source_path']}): {chunk['text'][:150]}...")

retrieved_chunks_2 = asyncio.run(retrieve_chunks(sample_query_2))
print(f"\nResults for '{sample_query_2}':")
for i, chunk in enumerate(retrieved_chunks_2):
    print(f"  Chunk {i+1} (ID: {chunk['id']}, Source: {chunk['source_path']}): {chunk['text'][:150]}...")

retrieved_chunks_3 = asyncio.run(retrieve_chunks(sample_query_3))
print(f"\nResults for '{sample_query_3}':")
for i, chunk in enumerate(retrieved_chunks_3):
    print(f"  Chunk {i+1} (ID: {chunk['id']}, Source: {chunk['source_path']}): {chunk['text'][:150]}...")


print("\n--- RAG Pipeline (Phase 1 & Retrieval) Complete! ---")
print(f"The LanceDB database is located at: {os.path.abspath(LANCEDB_PATH)}")
print("Next, you would feed these retrieved chunks as context to a Large Language Model (LLM) along with the user's query to generate a comprehensive answer.")
