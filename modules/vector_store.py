import faiss
import numpy as np
import os

VECTOR_STORE_PATH = "vector_store"
INDEX_FILE = os.path.join(VECTOR_STORE_PATH, "index.faiss")

def build_faiss_index(embeddings: np.ndarray):
    """Create FAISS index using cosine similarity (inner product)."""
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    return index

def load_faiss_index():
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)
    return None
