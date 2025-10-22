from chonkie import TokenChunker
from tokenizers import Tokenizer # You'll likely need a tokenizer, e.g., for GPT2

# Initialize a tokenizer (e.g., GPT-2 tokenizer)
# For simplicity, Chonkie's TokenChunker can often default to a GPT2 tokenizer if not specified.
# However, for more control or specific models, you might initialize it explicitly.
# You might need to install 'tokenizers' library: pip install tokenizers
try:
    tokenizer = Tokenizer.from_pretrained("gpt2")
except Exception:
    # Fallback if 'tokenizers' library or model isn't readily available
    print("Could not load 'gpt2' tokenizer. Using default Chonkie tokenizer.")
    tokenizer = None # Chonkie TokenChunker can often handle this internally

# Initialize the chunker
if tokenizer:
    chunker = TokenChunker(tokenizer=tokenizer)
else:
    chunker = TokenChunker() # Chonkie will try to use a default

# Your text to chunk
# text = """
# Chonkie is an open-source Python library designed to simplify text chunking for AI applications, especially for Retrieval-Augmented Generation (RAG) systems. It helps break down large documents into smaller, meaningful pieces. This process improves the efficiency and accuracy of large language models by ensuring they receive relevant and well-structured context within their token limits. Chonkie offers various chunking strategies like token-based, sentence-based, recursive, and semantic chunking.
# """

text = ""
with open('pydantic_docs_general/latest/concepts/types.md', 'r') as file:
    text = file.read()
    # print(text)

# Chunk the text
chunks = chunker(text)

# Display the chunks
print("--- Token Chunker Example ---")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    # print(f"  Text: {chunk.text}")
    print(f"  Tokens: {chunk.token_count}")
    print("-" * 20)