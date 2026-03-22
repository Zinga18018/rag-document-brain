import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """splits text into overlapping word-level chunks for embedding.

    uses a sliding window approach -- each chunk overlaps with the
    previous one by `overlap` words so we don't lose context at
    chunk boundaries.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def chunk(self, text: str) -> list[str]:
        """split a block of text into overlapping chunks."""
        words = text.split()
        if not words:
            return []

        chunks = []
        for i in range(0, len(words), self.step):
            segment = " ".join(words[i : i + self.chunk_size])
            if segment.strip():
                chunks.append(segment)

        logger.debug("chunked %d words into %d segments", len(words), len(chunks))
        return chunks

    @staticmethod
    def extract_text(content: bytes) -> str:
        """best-effort decode of raw file bytes into a string."""
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")
