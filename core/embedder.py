import logging

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """encodes text into dense vectors using a sentence-transformer model.

    auto-detects GPU availability and moves the model accordingly.
    the same instance handles both batch encoding and single-query encoding.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("loading %s on %s", model_name, self.device)
        self._model = SentenceTransformer(model_name, device=self.device)
        logger.info("embedder ready (dim=%d)", self.dimension)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """encode a list of strings into embedding vectors."""
        return self._model.encode(
            texts, show_progress_bar=False, batch_size=batch_size,
        ).tolist()

    def encode_query(self, query: str) -> list[float]:
        """encode a single query string."""
        return self.encode([query])[0]
