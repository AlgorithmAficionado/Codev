import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class DocumentChunk:
    """
    Represents a chunk of a document with metadata, text, and embedding.
    """
    def __init__(self, id_, chunk_id, text='', embedding=None, metadata=None):
        self.id_ = id_
        self.chunk_id = chunk_id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}

# Initialize the SentenceTransformer model globally
model = SentenceTransformer('all-MiniLM-L6-v2')


def chunk_text_with_overlap(text, chunk_size=100, overlap=20):
    """
    Splits text into overlapping chunks for better context preservation.
    Args:
        text (str): Input text to be chunked.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.
    Returns:
        list: List of overlapping chunks.
    """
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]


def process_documents(documents):
    """
    Process documents into chunks with embeddings.
    Args:
        documents (list of str): List of documents.
    Returns:
        list: List of DocumentChunk objects.
    """
    chunks = []
    for i, text in enumerate(documents):
        text_chunks = chunk_text_with_overlap(text)
        for j, chunk in enumerate(text_chunks):
            chunks.append(DocumentChunk(id_=f"{i}_{j}", chunk_id=j, text=chunk))
    return chunks


def compute_embeddings(chunks):
    """
    Compute embeddings for all chunks and assign to chunks.
    Args:
        chunks (list of DocumentChunk): List of DocumentChunk objects.
    Returns:
        np.ndarray: Array of embeddings.
    """
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
    return np.array(embeddings)


def create_faiss_index(embeddings):
    """
    Create a FAISS index from embeddings.
    Args:
        embeddings (np.ndarray): Array of embeddings.
    Returns:
        faiss.IndexFlatL2: FAISS index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search_index(index, query, chunks, top_k=5):
    """
    Search the FAISS index for the most relevant chunks to the query.
    Args:
        index (faiss.IndexFlatL2): FAISS index.
        query (str): Query string.
        chunks (list of DocumentChunk): List of DocumentChunk objects.
        top_k (int): Number of top results to retrieve.
    Returns:
        list: List of relevant chunks with scores.
    """
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:  # Check for valid index
            results.append({
                "text": chunks[idx].text,
                "metadata": chunks[idx].metadata,
                "distance": dist
            })
    return results


def get_relevant_context(documents, query, chunk_size=100, overlap=20, top_k=5):
    """
    Main function to process documents, build FAISS index, and retrieve relevant context for a query.
    Args:
        documents (list of str): List of documents.
        query (str): Query string.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between chunks.
        top_k (int): Number of top results to retrieve.
    Returns:
        list: List of relevant chunks with scores.
    """
    # Step 1: Process documents into chunks
    chunks = process_documents(documents)

    # Step 2: Compute embeddings for the chunks
    embeddings = compute_embeddings(chunks)

    # Step 3: Create FAISS index
    index = create_faiss_index(embeddings)

    # Step 4: Perform search on the query
    results = search_index(index, query, chunks, top_k)

    return results

# if __name__ == "__main__":
#     # Example documents
#     documents = [
#         "This is a document about machine learning and artificial intelligence.",
#         "Deep learning is a subset of machine learning focused on neural networks.",
#         "Transformers have revolutionized natural language processing.",
#     ]

#     # Query to search for
#     query = "Tell me about neural networks."

#     # Get relevant context
#     relevant_context = get_relevant_context(documents, query)

#     # Display results
#     print("Relevant Context:")
#     for idx, result in enumerate(relevant_context, 1):
#         print(f"{idx}. Text: {result['text']}")
#         print(f"   Distance: {result['distance']}")
