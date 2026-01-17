# 📝 Full Naïve RAG Pipeline (Gemini 2.5 Flash)

This project demonstrates a **Naïve Retrieval-Augmented Generation (RAG) pipeline** using the Gemini 2.5 Flash LLM (OpenAI-compatible). The pipeline extracts text from a PDF, chunks it, converts chunks into embeddings, stores them in FAISS, and retrieves relevant context to answer user queries.

---

## ✅ Key Steps in the Pipeline

1. **Load Environment Variables**

   * The pipeline uses a `.env` file to store the `GOOGLE_API_KEY` for the Gemini API.
   * Ensure your `.env` file contains:

     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

2. **PDF → Plain Text Extraction**

   * Input PDF files are read using PyMuPDF.
   * All pages are converted into plain text for further processing.

3. **Text Chunking**

   * The extracted text is split into **sentence-based chunks**.
   * Each chunk has a maximum character limit (e.g., 700 chars).
   * Chunks are saved with start/end delimiters to a text file.

4. **Chunk Embeddings & FAISS Index**

   * Each chunk is converted into a **vector embedding** using `SentenceTransformer` (model: `all-MiniLM-L6-v2`).
   * Embeddings are normalized for cosine similarity.
   * Chunks are stored in a **FAISS index** for fast similarity search.

5. **User Query → Embedding**

   * Users input a question via the terminal.
   * The query is converted into an embedding vector using the same model.

6. **Vector Search**

   * FAISS retrieves the **top-k most similar chunks** based on cosine similarity.
   * These chunks provide the context for the LLM.

7. **Prompt Construction**

   * Retrieved chunks are combined into a single context.
   * A clear instruction is prepended to guide the LLM to answer using only the provided context.

8. **Gemini 2.5 Flash LLM Call**

   * The constructed prompt is sent to the Gemini model via OpenAI-compatible API.
   * The LLM returns the final answer, restricted to the content of the retrieved chunks.

---

## ⚡ How to Use

1. **Install Dependencies**

   ```bash
   pip install fitz PyMuPDF nltk sentence-transformers faiss-cpu openai python-dotenv
   ```

2. **Create a `.env` File**

   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

3. **Place Your PDF**

   * Example: `Policy_Highlighted.pdf` in the project folder.

4. **Run the Pipeline**

   ```bash
   python naive_rag_pipeline.py
   ```

5. **Enter Your Question**

   * Type your query in the terminal.
   * The system retrieves relevant chunks and provides an answer.

6. **View Output**

   * Chunked text is saved to `chunks.txt`.
   * LLM output is printed in the terminal.

---
## Summary (With Examples)

### Embedding

Convert text into numbers that represent its meaning

```text
"I love cats" → [0.12, -0.44, 0.98, ...]
```

(model: all-MiniLM-L6-v2)

### Semantic Search

Find information based on meaning, not exact words

```text
Search: "cats as pets"
Finds: "I love cats" ✅
```

(using cosine similarity)

### Cosine Similarity

A way to measure how close two meanings are

```python
similarity = cosine_similarity(embedding1, embedding2)
```

Higher value → meanings are more similar

### Vector Database

Stores number vectors to quickly find the closest matches

* Examples: FAISS, Chroma, Pinecone

Workflow:

```
Store embeddings → search nearest vectors → get best match
```

---

## ⚠️ Notes

* Ensure the Gemini API key is valid.
* FAISS stores vectors in memory; for large PDFs, consider persistent vector DBs.
* Sentence-based chunking ensures semantic context is preserved.
* This pipeline is a **Naïve RAG** approach and may be enhanced with overlapping chunks or hybrid methods for higher recall.
