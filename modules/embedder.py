from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
from typing import List

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: List[str]) -> np.ndarray:
    embeddings = _model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return normalize(embeddings, axis=1)
