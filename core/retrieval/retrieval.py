from core.embeddings.embedder import Embedder
from core.vector_store.lancedb import LanceDBManager

class Retriever:
    """
    Handles the retrieval of relevant chunks from the vector store based on a query.
    """
    def __init__(self, embedder: Embedder, db_manager: LanceDBManager):
        """
        Initializes the Retriever with an Embedder and a LanceDBManager instance.

        Args:
            embedder (Embedder): An instance of the Embedder class.
            db_manager (LanceDBManager): An instance of the LanceDBManager class.
        """
        self.embedder = embedder
        self.db_manager = db_manager

    async def retrieve(self, query_text: str, k: int = 3):
        """
        Retrieves the top-k most relevant chunks from LanceDB for a given query.

        Args:
            query_text (str): The user's query.
            k (int): The number of top relevant chunks to retrieve.

        Returns:
            list: A list of dictionaries, where each dictionary represents a retrieved chunk.
        """
        print(f"\nRetrieving for query: '{query_text}'")
        # 1. Embed the query
        query_embeddings_list = await self.embedder.get_embeddings([query_text])

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
            table = await self.db_manager.get_table()
            if table is None:
                print("LanceDB table not available for retrieval.")
                return []
            
            results = table.search(query_vector).limit(k).to_list()
            print(f"Found {len(results)} relevant chunks.")
            return results
        except Exception as e:
            print(f"Error during LanceDB retrieval: {e}")
            return []
