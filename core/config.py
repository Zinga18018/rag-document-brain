from dataclasses import dataclass


@dataclass
class RAGConfig:
    """central place for all RAG pipeline settings."""

    embed_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500           # words per chunk
    chunk_overlap: int = 50         # overlapping words between chunks
    collection_name: str = "documents"
    similarity_metric: str = "cosine"
    default_top_k: int = 5
    max_file_size_mb: int = 10
    port: int = 8002
