from fastapi import File, HTTPException, UploadFile

from .schemas import (
    QueryRequest, TextIngestRequest,
    IngestResponse, QueryResponse,
)


def register_routes(app, pipeline):
    """attach all API endpoints to the FastAPI app."""

    @app.get("/health")
    async def health():
        info = pipeline.status()
        return {"status": "healthy", **info}

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest_file(file: UploadFile = File(...)):
        content = await file.read()
        if not content.strip():
            raise HTTPException(400, "empty or unreadable file")
        result = pipeline.ingest_bytes(content, file.filename)
        return IngestResponse(**result)

    @app.post("/ingest/text")
    async def ingest_text(req: TextIngestRequest):
        if not req.text.strip():
            raise HTTPException(400, "empty text")
        return pipeline.ingest_text(req.text, req.title)

    @app.post("/query", response_model=QueryResponse)
    async def query_documents(req: QueryRequest):
        if pipeline.store.count == 0:
            raise HTTPException(400, "no documents ingested yet")
        result = pipeline.query(req.question, req.top_k)
        return QueryResponse(**result)

    @app.get("/documents")
    async def list_documents():
        return pipeline.status()

    @app.delete("/documents")
    async def clear_documents():
        pipeline.store.clear()
        return {"status": "cleared", "total_chunks": 0}

    @app.post("/similar")
    async def find_similar(body: dict):
        text = body.get("text", "")
        top_k = body.get("top_k", 5)
        if not text.strip():
            raise HTTPException(400, "empty text")
        q_vec = pipeline.embedder.encode_query(text)
        return {"results": pipeline.store.search(q_vec, top_k)}
