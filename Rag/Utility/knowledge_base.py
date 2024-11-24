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
    """
    if not text.strip():
        return []  # Return an empty list for empty or whitespace text
    
    words = text.split()
    if chunk_size <= 0 or overlap < 0 or chunk_size <= overlap:
        chunk_size, overlap = 100, 20  # Reset to default values if parameters are invalid
    
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]


def process_documents(documents):
    """
    Process documents into chunks with embeddings.
    """
    if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
        documents = []  # Ensure documents is a list of strings
    
    chunks = []
    for i, text in enumerate(documents):
        if not text.strip():
            continue  # Skip empty or whitespace-only documents
        
        text_chunks = chunk_text_with_overlap(text)
        for j, chunk in enumerate(text_chunks):
            chunks.append(DocumentChunk(id_=f"{i}_{j}", chunk_id=j, text=chunk))
    
    return chunks


def compute_embeddings(chunks):
    """
    Compute embeddings for all chunks and assign to chunks.
    """
    if not chunks:
        # Return an empty array for empty chunks
        return np.array([])
    
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
    return np.array(embeddings)


def create_faiss_index(embeddings):
    """
    Create a FAISS index from embeddings.
    """
    if embeddings.size == 0:
        # Return an empty FAISS index for empty embeddings
        return faiss.IndexFlatL2(1)  # Create a dummy index with 1 dimension
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search_index(index, query, chunks, top_k=5):
    """
    Search the FAISS index for the most relevant chunks to the query.
    """
    if not query.strip():
        return []  # Return an empty list for an empty query
    
    query_embedding = model.encode([query], normalize_embeddings=True)
    if index.ntotal == 0:
        return []  # Return an empty list if the FAISS index has no data
    
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

# Example usage
# documents = []  # This can be an empty list
# query = "Tell me about neural networks."  # Query can be any string
# relevant_context = get_relevant_context(documents, query)
# print(relevant_context)
