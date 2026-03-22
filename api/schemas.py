from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class TextIngestRequest(BaseModel):
    text: str
    title: str = "untitled"


class IngestResponse(BaseModel):
    source: str
    chunks_created: int
    total_chars: int
    ingest_ms: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query_ms: float
    total_chunks: int
