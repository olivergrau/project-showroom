import re
from typing import List, Dict


class TextChunker:
    """Utility class for splitting text into overlapping chunks."""

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[Dict]:
        """Return a list of chunk dictionaries without writing to disk."""
        text = re.sub(r"\s+", " ", text).strip()
        separator = "\n"
        step = self.chunk_size - self.chunk_overlap

        chunks = []
        start = 0
        chunk_id = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            window_text = text[start:end]
            last_sep = window_text.rfind(separator)
            if last_sep != -1 and (start + last_sep + len(separator)) < text_len:
                end = start + last_sep + len(separator)

            chunk_text = text[start:end]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_size": len(chunk_text),
                    "start_char": start,
                    "end_char": end,
                }
            )

            start += step
            chunk_id += 1

        return chunks
