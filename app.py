import streamlit as st
import tempfile
import os
from modules import parser_chunker, embedder, retriever
import pandas as pd
import numpy as np

st.set_page_config(page_title="📄 PDF Question Answering", layout="wide")
st.title("📄 PDF Question Answering")

# --- Session state ---
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None and not st.session_state.pdf_uploaded:
    st.info("📄 Processing uploaded PDF...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    raw_text = parser_chunker.extract_text_from_pdf(tmp_path)
    chunks = parser_chunker.chunk_text(raw_text)
    embeddings = retriever.embed_text(chunks)
    norm_embeddings = np.array([e / np.linalg.norm(e) for e in embeddings])
    index = retriever.create_index(norm_embeddings)

    st.session_state.chunks = chunks
    st.session_state.embeddings = norm_embeddings
    st.session_state.index = index
    st.session_state.pdf_uploaded = True

    os.remove(tmp_path)
    st.success(f"✅ PDF processed and {len(chunks)} chunks stored.")

elif st.session_state.pdf_uploaded:
    st.info("✓ PDF already processed.")

# --- Ask Question ---
st.markdown("### 🔍 Ask a question about the PDF")
user_query = st.text_input("Enter your question")

if user_query:
    top_k = 3
    indices, distances = retriever.retrieve_chunks(
        user_query,
        st.session_state.index,
        top_k=top_k
    )
    retrieved_chunks = [st.session_state.chunks[idx] for idx in indices]

    # --- LLM Answer ---
    st.markdown("#### 💬 Answer from Local LLM:")
    with st.spinner("Generating answer..."):
        answer = retriever.generate_answer(user_query, retrieved_chunks)
        st.success(answer)

    # --- Feedback ---
    feedback = None
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Satisfied", key="thumbs_up"):
            feedback = "👍"
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 Needs work", key="thumbs_down"):
            feedback = "👎"
            st.warning("Thanks — we'll log this!")

    if feedback:
        st.session_state.qa_log.append({
            "query": user_query,
            "answer": answer,
            "chunks": indices,
            "feedback": feedback
        })

    # --- Retrieved Chunks ---
    with st.expander("📎 Retrieved Chunks (Top Matches)", expanded=False):
        for idx, score in zip(indices, distances):
            st.markdown(f"**Chunk #{idx}** — Score: `{score:.2f}`")
            st.code(st.session_state.chunks[idx], language="markdown")

# --- Developer Panel ---
with st.expander("🛠️ Developer Options", expanded=False):
    st.markdown("**Embedding Preview (First 10 dims of first chunk)**")
    if st.session_state.embeddings is not None:
        preview = np.round(st.session_state.embeddings[0][:10], 4)
        st.text_area("Embedding", str(preview), height=80)

    st.markdown("**Q&A Log (including feedback)**")
    if st.session_state.qa_log:
        df = pd.DataFrame(st.session_state.qa_log)
        st.dataframe(df)
