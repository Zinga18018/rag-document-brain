# RAG Document Brain

A semantic passage-search API and Streamlit interface. It splits text into overlapping chunks, embeds them with `all-MiniLM-L6-v2` and retrieves similar passages from Chroma. Despite the project name, this implementation does not load an answer-generating LLM: the response's `answer` field formats retrieved previews and source filenames.

## Retrieval flow

```text
text → 500-word chunks, 50-word overlap → MiniLM embeddings → Chroma
query → MiniLM embedding → cosine search → passage previews + sources
```

[core/pipeline.py](core/pipeline.py) coordinates ingestion and retrieval. [core/chunker.py](core/chunker.py) decodes uploaded bytes as UTF-8, falling back to Latin-1. It does not parse PDFs, even though the upload widget accepts them; CSV input is treated as raw text, not a table. Use plain-text files or `/ingest/text`.

## Run locally

Use Python 3.10+. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8002
```

For Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead. Open `http://127.0.0.1:8002/docs`. Importing the application initializes the embedding model, which downloads if not cached. Dependencies are unpinned.

For the separate upload/search interface, run `streamlit run app.py`. It creates its own pipeline, not an API client, so its store is not the API's store.

## API

| Method | Path | Behavior |
|---|---|---|
| GET | `/health` | Pipeline and store status |
| POST | `/ingest` | Decode and ingest uploaded bytes |
| POST | `/ingest/text` | Ingest text with a source name |
| POST | `/query` | Retrieve passage previews and sources |
| GET | `/documents` | List ingested sources |
| DELETE | `/documents` | Clear the store |
| POST | `/similar` | Retrieve passages similar to input text |

## Current limits

- [core/vectorstore.py](core/vectorstore.py) uses an ephemeral Chroma client, not persistent storage.
- Similarity scores are not probabilities that a passage answers the question. Chunk overlap does not guarantee that all context is retained.
- The configured 10 MB file-size setting is not enforced by the upload route. A selected Streamlit file can be ingested again on reruns, producing duplicate chunks.
- The API has no authentication, including on store deletion, and permits all CORS origins. The [loopback launch](https://uvicorn.dev/settings/) above is for local use; `python main.py` instead binds all interfaces with reload enabled.
- PDF extraction, generated answers, retrieval quality and claimed sub-millisecond performance are not established by this implementation. No tests or benchmark results are committed.

Setup, model execution and browser flows were not rerun for this documentation update.

## Related implementations

[ML-Portfolio-Projects](https://github.com/Zinga18018/ML-Portfolio-Projects) contains an [retrieval API variant](https://github.com/Zinga18018/ML-Portfolio-Projects/tree/master/project2-rag-document-brain) and a different [HF Space variant](https://github.com/Zinga18018/ML-Portfolio-Projects/tree/master/hf-spaces/rag-document-brain). The HF variant adds TinyLlama answer generation and different document handling. Those features should not be assumed to exist in this repository.

