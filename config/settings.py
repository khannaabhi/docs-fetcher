import os

# Define the path for your LanceDB database
LANCEDB_PATH = "lancedb_rag_data"
TABLE_NAME = "documents"

# --- Local Embedding Model Configuration ---
# Set to True to use local embeddings, False to use OpenAI embeddings
USE_LOCAL_EMBEDDINGS = True
LOCAL_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A good default small model

# You can add more configuration variables here as your project grows
