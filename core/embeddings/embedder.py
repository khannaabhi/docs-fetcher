import asyncio
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Manages different embedding models (OpenAI and local SentenceTransformers).
    Provides a unified interface to generate embeddings.
    """
    def __init__(self, use_local: bool, local_model_name: str = "all-MiniLM-L6-v2", openai_api_key: str = None):
        """
        Initializes the Embedder.

        Args:
            use_local (bool): If True, uses a local SentenceTransformer model.
                              If False, uses OpenAI embeddings.
            local_model_name (str): Name of the local SentenceTransformer model to load.
            openai_api_key (str): OpenAI API key for OpenAI embeddings.
        """
        self.use_local = use_local
        self.local_model_name = local_model_name
        self.openai_api_key = openai_api_key
        self.local_embedding_model = None
        self.openai_client = None

        if not self.use_local and not self.openai_api_key:
            print("WARNING: OpenAI API key not provided. OpenAI embeddings might not work.")

    async def load_model(self):
        """
        Loads the appropriate embedding model (local or initializes OpenAI client).
        This should be called once at the start of the application.
        """
        if self.use_local:
            try:
                print(f"Loading local embedding model: {self.local_model_name}...")
                self.local_embedding_model = SentenceTransformer(self.local_model_name)
                print("Local embedding model loaded successfully.")
            except Exception as e:
                print(f"Error loading local embedding model: {e}")
                print("Falling back to OpenAI embeddings if available (if API key is set).")
                self.use_local = False # Disable local embeddings if loading fails
        
        if not self.use_local and self.openai_api_key:
            print("Initializing OpenAI client.")
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        elif not self.use_local and not self.openai_api_key:
            print("No embedding model initialized. Ensure an API key is set or local model loads.")


    async def _get_openai_embeddings_async(self, texts: list[str]):
        """
        Asynchronously gets embeddings for a list of texts using OpenAI's API.
        """
        if not self.openai_client:
            print("OpenAI client not initialized. Cannot generate OpenAI embeddings.")
            return [None] * len(texts)
        try:
            response = await self.openai_client.embeddings.create(
                input=texts,
                model="text-embedding-3-small" # Or "text-embedding-3-large" for higher quality
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error generating OpenAI embeddings: {e}")
            return [None] * len(texts)

    async def _get_local_embeddings_async(self, texts: list[str]):
        """
        Asynchronously gets embeddings for a list of texts using a local SentenceTransformer model.
        """
        if self.local_embedding_model is None:
            print("Local embedding model is not loaded. Cannot generate local embeddings.")
            return [None] * len(texts)
        try:
            # SentenceTransformer encode method is synchronous, but we keep the async wrapper
            # for consistency and potential future use with run_in_executor for large batches.
            embeddings = self.local_embedding_model.encode(texts, convert_to_list=True)
            return embeddings
        except Exception as e:
            print(f"Error generating local embeddings: {e}")
            return [None] * len(texts)

    async def get_embeddings(self, texts: list[str]):
        """
        Generates embeddings for a list of texts using the configured model.

        Args:
            texts (list[str]): A list of strings to embed.

        Returns:
            list: A list of embedding vectors (list of floats), or None for failed embeddings.
        """
        if self.use_local:
            print(f"  Using local embedding model: {self.local_model_name}")
            return await self._get_local_embeddings_async(texts)
        else:
            print("  Using OpenAI embedding model.")
            return await self._get_openai_embeddings_async(texts)

