# ----------------------------
# Full Naïve RAG Pipeline
# Gemini 2.5 Flash (OpenAI-compatible)
# ----------------------------

import os
import fitz  # PyMuPDF
import nltk
import faiss
import numpy as np

from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ----------------------------
# 0️⃣ Load Environment Variables
# ----------------------------
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY not found in .env")

# ----------------------------
# CONFIGURATION
# ----------------------------
PDF_PATH = "Policy_Highlighted.pdf"
CHUNK_TXT_PATH = "chunks.txt"
TOP_K = 3
MODEL_NAME = "gemini-2.5-flash"

# OpenAI-compatible Gemini client
client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ----------------------------
# 1️⃣ PDF → Plain Text
# ----------------------------
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

text = extract_pdf_text(PDF_PATH)
print("✅ PDF text extracted.")

# ----------------------------
# 2️⃣ Sentence Chunking
# ----------------------------
nltk.download("punkt", quiet=True)

def sentence_chunk(text, max_chars=700):
    chunks, current = [], ""
    for sent in sent_tokenize(text):
        if len(current) + len(sent) <= max_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks

chunks = sentence_chunk(text)

with open(CHUNK_TXT_PATH, "w", encoding="utf-8") as f:
    for idx, chunk in enumerate(chunks, start=1):
        f.write(f"--- CHUNK {idx} START ---\n")
        f.write(chunk + "\n")
        f.write(f"--- CHUNK {idx} END ---\n\n")

print(f"✅ {len(chunks)} chunks created and saved to chunks.txt with start/end delimiters")


# ----------------------------
# 3️⃣ Embeddings → FAISS
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = embed_model.encode(
    chunks,
    convert_to_numpy=True,
    show_progress_bar=False
)

# Normalize for cosine similarity
chunk_embeddings = chunk_embeddings / np.linalg.norm(
    chunk_embeddings, axis=1, keepdims=True
)

dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(chunk_embeddings)

print("✅ Chunk embeddings stored in FAISS.")

# ----------------------------
# 4️⃣ User Query → Embedding
# ----------------------------
user_query = input("\nEnter your question: ")

query_embedding = embed_model.encode(
    [user_query],
    convert_to_numpy=True
)

query_embedding = query_embedding / np.linalg.norm(
    query_embedding, axis=1, keepdims=True
)

print("\n✅ Query embedding (first 10 dims):")
print(query_embedding[0][:10])

# ----------------------------
# 5️⃣ Vector Search (Cosine Similarity)
# ----------------------------
scores, indices = index.search(query_embedding, TOP_K)

retrieved_chunks = [chunks[i] for i in indices[0]]
retrieved_embeddings = [chunk_embeddings[i] for i in indices[0]]

print("\n✅ Retrieved Chunks sent to LLM:")
for i in indices[0]:
    print(f"Chunk {i+1}")

# ----------------------------
# 6️⃣ Prompt Construction
# ----------------------------
instruction = (
    "Answer the question ONLY using the provided context. "
    "If the answer is not found, say 'Not found in document.'"
)

context = "\n\n".join(retrieved_chunks)

llm_prompt = f"""
{instruction}

CONTEXT:
{context}

QUESTION:
{user_query}
"""

# ----------------------------
# 7️⃣ Gemini 2.5 Flash Call
# ----------------------------
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": llm_prompt}
    ],
    temperature=0,
    max_tokens=500
)

answer = response.choices[0].message.content

print("\n✅ Answer from Gemini:")
print(answer)
