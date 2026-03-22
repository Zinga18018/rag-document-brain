import uuid
import logging

import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    """wraps ChromaDB to handle document storage + cosine similarity search.

    each document chunk gets a unique ID, source attribution, and its
    position within the original file so we can reconstruct context later.
    """

    def __init__(self, collection_name: str = "documents",
                 similarity_metric: str = "cosine"):
        self._client = chromadb.Client()
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": similarity_metric},
        )
        logger.info("vectorstore initialized (collection=%s)", collection_name)

    @property
    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[str], embeddings: list[list[float]], source: str):
        """insert chunked text with their embeddings into the store."""
        ids = [f"{source}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [
            {"filename": source, "chunk_idx": i, "chunk_total": len(chunks)}
            for i in range(len(chunks))
        ]
        self._collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info("added %d chunks from '%s'", len(chunks), source)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """return the top-k most similar chunks with similarity scores."""
        n = min(top_k, max(self.count, 1))
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "filename": results["metadatas"][0][i].get("filename", "unknown"),
                "chunk_idx": results["metadatas"][0][i].get("chunk_idx", 0),
                "similarity": round(1 - results["distances"][0][i], 4),
            })
        return hits

    def list_sources(self) -> list[str]:
        """return a sorted list of unique source filenames."""
        if self.count == 0:
            return []
        all_data = self._collection.get(include=["metadatas"])
        return sorted({m.get("filename", "unknown") for m in all_data["metadatas"]})

    def clear(self):
        """drop all documents and recreate the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vectorstore cleared")
