import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text_list):
    return _model.encode(text_list)

def create_index(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatIP(dim)
    norm_embeddings = np.array([e / np.linalg.norm(e) for e in embeddings])
    index.add(norm_embeddings)
    return index

def retrieve_chunks(query, index, top_k=3):
    query_embedding = embed_text([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    scores, indices = index.search(query_embedding, top_k)
    return indices[0].tolist(), scores[0].tolist()

def generate_answer(query, context_chunks, model="mistral"):
    system_prompt = (
        "You are a helpful assistant answering questions based on a document. "
        "Answer only from the provided context. If unsure, say you don't know.\n\n"
    )
    context = "\n\n".join(context_chunks)
    prompt = f"{system_prompt}Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"❌ Error querying local model: {e}"
