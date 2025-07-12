import pdfplumber
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text

def chunk_text(text, max_sentences_per_chunk=3):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    for i in range(0, len(sentences), max_sentences_per_chunk):
        chunk = " ".join(sentences[i:i + max_sentences_per_chunk])
        chunks.append(chunk)
    return chunks
