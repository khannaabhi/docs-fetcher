import os
from dotenv import load_dotenv

# Import components from their new locations
from config.settings import (
    LANCEDB_PATH, TABLE_NAME, USE_LOCAL_EMBEDDINGS, LOCAL_EMBEDDING_MODEL_NAME
)
from core.data_ingestion.data_loader import populate_dummy_data, read_markdown_files
from core.chunker.chunker import Chunker
from core.embeddings.embedder import Embedder
from core.vector_store.lancedb import LanceDBManager
from core.retrieval.retrieval import Retriever


# --- Main Orchestrator Function ---
async def run():
    """
    Orchestrates the entire RAG pipeline from data loading to retrieval.
    """
    load_dotenv() # Load environment variables from .env file

    # 1. Initialize components
    embedder, chunker, db_manager, retriever = await initialize_components()

    # # 2. Load data
    DATA_DIR = "temp_docs/istio_docs_general"
    raw_documents = load_data(DATA_DIR)

    # # 3. Process and Store data
    await process_and_store_data(raw_documents, chunker, embedder, db_manager)

    # 4. Query and Retrieve
    # await query_and_retrieve(retriever)

    print("\n--- RAG Pipeline Orchestration Complete! ---")
    print(f"The LanceDB database is located at: {os.path.abspath(LANCEDB_PATH)}")
    print("Next, you would feed these retrieved chunks as context to a Large Language Model (LLM) along with the user's query to generate a comprehensive answer.")



# --- 1. Initialization Phase ---
async def initialize_components():
    """
    Initializes and returns instances of the core RAG pipeline components.
    """
    print("--- Initializing RAG Pipeline Components ---")

    # Initialize Embedder
    embedder = Embedder(
        use_local=USE_LOCAL_EMBEDDINGS,
        local_model_name=LOCAL_EMBEDDING_MODEL_NAME,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    await embedder.load_model() # Load local model or initialize OpenAI client

    # Initialize Chunker
    chunker = Chunker(min_chunk_size=50, chunk_size=200)

    # Initialize LanceDB Manager
    db_manager = LanceDBManager(db_path=LANCEDB_PATH, table_name=TABLE_NAME)

    # Initialize Retriever
    retriever = Retriever(embedder=embedder, db_manager=db_manager)

    print("Components initialized successfully.")
    return embedder, chunker, db_manager, retriever

# --- 2. Loader Phase ---
def load_data(data_dir: str):
    """
    Loads raw data from the specified directory.
    Includes populating dummy data and reading markdown files.

    Args:
        data_dir (str): The directory containing the markdown files.

    Returns:
        list: A list of dictionaries, each representing a raw document.
    """
    print(f"\n--- Loading Data from '{data_dir}' ---")
    # Populate dummy data and clean up old LanceDB data for a fresh start
    populate_dummy_data(data_dir=data_dir, lancedb_path=LANCEDB_PATH)
    print(f"LanceDB will be stored at: {LANCEDB_PATH}")

    raw_documents = read_markdown_files(data_dir)
    print(f"Found {len(raw_documents)} Markdown files.")
    return raw_documents

# --- 3. Store Phase (Generate Embeddings and Store) ---
async def process_and_store_data(
    raw_documents: list[dict],
    chunker: Chunker,
    embedder: Embedder,
    db_manager: LanceDBManager
):
    """
    Processes raw documents by chunking, generating embeddings, and storing them in LanceDB.

    Args:
        raw_documents (list[dict]): List of raw documents to process.
        chunker (Chunker): An instance of the Chunker.
        embedder (Embedder): An instance of the Embedder.
        db_manager (LanceDBManager): An instance of the LanceDBManager.
    """
    print("\n--- Processing and Storing Data ---")
    all_documents_to_store = []
    for doc_idx, raw_doc in enumerate(raw_documents):
        doc_text = raw_doc["text"]
        doc_source_path = raw_doc["source_path"]

        print(f"\nProcessing document from: {doc_source_path}")
        chunks = chunker.chunk_document(doc_text) # Call chunker's method
        print(f"  Generated {len(chunks)} chunks for this document.")

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await embedder.get_embeddings(chunk_texts) # Call embedder's method

        for i, chunk in enumerate(chunks):
            if embeddings[i] is not None:
                doc_id = f"{doc_source_path}_{i}" # Unique ID for each chunk
                all_documents_to_store.append({
                    "id": doc_id,
                    "text": chunk.text,
                    "source_path": doc_source_path,
                    "vector": embeddings[i]
                })
            else:
                print(f"  Skipping chunk {i} from {doc_source_path} due to embedding failure.")

    await db_manager.add_documents(all_documents_to_store) # Add all processed documents to LanceDB
    print(f"Total documents in LanceDB table '{TABLE_NAME}': {await db_manager.get_document_count()}")
    print("Data processing and storage complete.")

# --- 4. Query and Retriever Phase ---
async def query_and_retrieve(retriever: Retriever):
    """
    Performs sample queries and retrieves relevant chunks using the Retriever.

    Args:
        retriever (Retriever): An instance of the Retriever.
    """
    print("\n--- Starting Query and Retrieval Phase ---")
    sample_query_1 = "How to install Project A?"
    sample_query_2 = "What is Project B about and its technologies?"
    sample_query_3 = "Tell me about the real-time dashboard."

    print("\n--- Running Sample Queries ---")

    queries = [sample_query_1, sample_query_2, sample_query_3]

    queries = ["How does Istio provide traffic routing between microservices?"]

    for query in queries:
        retrieved_chunks = await retriever.retrieve(query, k=5) # Call retriever's method
        print(f"\nResults for '{query}':")
        if retrieved_chunks:
            for i, chunk in enumerate(retrieved_chunks):
                print(f"  Chunk {i+1} (ID: {chunk['id']}, Source: {chunk['source_path']}): {chunk['text'][:]}...")
        else:
            print("  No relevant chunks found.")

    print("\nQuery and retrieval complete.")
