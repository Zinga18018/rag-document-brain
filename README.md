# Retrieval-Augmented Generation for Document Q&A

A RAG pipeline that lets you upload documents (text, PDF, CSV), chunks and embeds them with Sentence-Transformers, stores vectors in ChromaDB, and answers questions by retrieving relevant passages and grounding responses with source citations.

## how it works

```
document → chunking (500w, 50w overlap) → embedding (384d) → ChromaDB
                                                                  ↓
question → embed query → cosine similarity search → top-k chunks → answer
```

the pipeline breaks documents into overlapping word-level chunks so nothing gets lost at boundaries. each chunk gets a 384-dimensional embedding from all-MiniLM-L6-v2, then goes into ChromaDB for fast similarity search. when you ask a question, it finds the most relevant chunks and returns them with similarity scores and source attribution.

**stack:**
- all-MiniLM-L6-v2 for embeddings (22M params, 384d vectors)
- ChromaDB with HNSW indexing for sub-millisecond retrieval
- FastAPI + Uvicorn for the REST API

## setup

```bash
pip install -r requirements.txt
python main.py
```

runs at `localhost:8002`. swagger docs at `/docs`.

## api

| endpoint | method | what it does |
|----------|--------|-------------|
| `/health` | GET | pipeline status + store stats |
| `/ingest` | POST | upload and ingest a file |
| `/ingest/text` | POST | ingest raw text with a source name |
| `/query` | POST | semantic search over your documents |
| `/documents` | GET | list all ingested sources |
| `/documents` | DELETE | clear the vector store |
| `/similar` | POST | find similar passages to input text |

## architecture

```
core/
├── config.py       → chunk sizes, model names, all settings
├── chunker.py      → word-level text splitting with overlap
├── embedder.py     → sentence-transformers wrapper
├── vectorstore.py  → ChromaDB operations + cosine search
└── pipeline.py     → orchestrates chunk → embed → store → query

api/
├── schemas.py      → pydantic request/response models
└── routes.py       → endpoint handlers

main.py             → FastAPI entry point
app.py              → streamlit frontend
```

## streamlit demo

```bash
streamlit run app.py
```

upload documents, ask questions, and see retrieved chunks with similarity scores.

## requirements

- python 3.10+
- GPU optional (embeddings are fast on CPU too)
