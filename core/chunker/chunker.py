from chonkie import SemanticChunker, Visualizer, RecursiveChunker

class Chunker:
    """
    Manages the chunking process using chonkie.ai's SemanticChunker.
    """
    def __init__(self, min_chunk_size: int = 50, chunk_size: int = 200):
        """
        Initializes the Chunker with specified chunking parameters.

        Args:
            min_chunk_size (int): Minimum tokens per chunk.
            chunk_size (int): Maximum tokens per chunk.
        """
        print(f"Initializing Chunker with min_chunk_size={min_chunk_size}, chunk_size={chunk_size}")
        self.chunker = SemanticChunker(
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
            # Note: Chonkie might use its own internal embedding model for semantic
            # calculations. This is separate from the embedding model used for LanceDB.
            # If you need to configure Chonkie's internal model, refer to its docs.
        )
        self.chunker = RecursiveChunker.from_recipe("markdown", lang="en")
        self.viz = Visualizer()

    def chunk_document(self, text: str):
        """
        Chunks a given text document into semantic chunks.

        Args:
            text (str): The input text document to be chunked.

        Returns:
            list: A list of SemanticChunk objects. Each object has 'text' and potentially 'id'.
        """
        print("  Chunking document...")
        chunks = self.chunker.chunk(text)
        self.viz.save("chonkie-new.html", chunks)
        return chunks
