from chonkie import SemanticChunker
from autotiktokenizer import AutoTikTokenizer
from chonkie.embeddings import SentenceTransformerEmbeddings # Import Chonkie's wrapper

# Initialize a tokenizer
tokenizer = AutoTikTokenizer.from_pretrained("gpt2")

# Initialize Chonkie's SentenceTransformerEmbeddings wrapper,
# passing the model name as a string
embedding_model_wrapper = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")

# Initialize the semantic chunker with the wrapped embedding model
chunker = SemanticChunker(
    tokenizer=tokenizer,
    embedding_model=embedding_model_wrapper, # Use the wrapped model here
    max_chunk_size=512,
)

# text_for_semantic = """
# Artificial intelligence (AI) is a rapidly developing field that is transforming various aspects of human life. From self-driving cars to medical diagnostics, AI is making significant strides. Machine learning, a subset of AI, focuses on enabling systems to learn from data without explicit programming. Deep learning, in turn, is a specialized area within machine learning that uses neural networks with many layers to analyze data. These advancements are leading to powerful applications in natural language processing, computer vision, and robotics.
# """
text_for_semantic = ""
with open('pydantic_docs_general/latest/concepts/types.md', 'r') as file:
    text_for_semantic = file.read()

# Chunk the text semantically
semantic_chunks = chunker(text_for_semantic)

print("\n--- Semantic Chunker Example ---")
for i, chunk in enumerate(semantic_chunks):
    print(f"Semantic Chunk {i+1}:")
    print(f"  Text: {chunk.text}")
    print(f"  Tokens: {chunk.token_count}")
    if hasattr(chunk, 'sentences'):
        for j, sentence in enumerate(chunk.sentences):
            print(f"    Sentence {j+1}: {sentence.text}")
    print("-" * 20)