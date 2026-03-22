import time
import logging

from .config import RAGConfig
from .chunker import TextChunker
from .embedder import Embedder
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """orchestrates the full retrieval-augmented generation workflow.

    ties together chunking, embedding, storage, and querying into
    a single interface. each stage is its own class so you can swap
    components without touching the pipeline logic.
    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.chunker = TextChunker(self.config.chunk_size, self.config.chunk_overlap)
        self.embedder = Embedder(self.config.embed_model)
        self.store = VectorStore(
            self.config.collection_name,
            self.config.similarity_metric,
        )

    def ingest_text(self, text: str, source: str = "untitled") -> dict:
        """chunk, embed, and store a block of text."""
        start = time.perf_counter()
        chunks = self.chunker.chunk(text)
        embeddings = self.embedder.encode(chunks)
        self.store.add(chunks, embeddings, source)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info("ingested '%s': %d chunks in %.0fms", source, len(chunks), elapsed)
        return {
            "source": source,
            "chunks_created": len(chunks),
            "total_chars": len(text),
            "ingest_ms": round(elapsed, 1),
        }

    def ingest_bytes(self, content: bytes, filename: str) -> dict:
        """decode raw bytes and ingest as text."""
        text = self.chunker.extract_text(content)
        return self.ingest_text(text, filename)

    def query(self, question: str, top_k: int = 5) -> dict:
        """embed a question and retrieve the most relevant chunks."""
        start = time.perf_counter()
        q_vec = self.embedder.encode_query(question)
        hits = self.store.search(q_vec, top_k)
        elapsed = (time.perf_counter() - start) * 1000

        # build a human-readable answer summary
        lines = [f"Found {len(hits)} relevant passages:\n"]
        for i, hit in enumerate(hits, 1):
            preview = hit["text"][:200] + ("..." if len(hit["text"]) > 200 else "")
            lines.append(
                f"[{i}] {hit['filename']} (relevance {hit['similarity']:.0%}):\n   {preview}\n"
            )

        return {
            "answer": "\n".join(lines),
            "sources": hits,
            "query_ms": round(elapsed, 1),
            "total_chunks": self.store.count,
        }

    def status(self) -> dict:
        return {
            "sources": self.store.list_sources(),
            "total_chunks": self.store.count,
            "embed_model": self.embedder.model_name,
            "embed_dim": self.embedder.dimension,
        }
