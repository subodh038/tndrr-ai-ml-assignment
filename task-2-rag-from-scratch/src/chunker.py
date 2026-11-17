"""
chunker.py - Text chunking strategies for the RAG pipeline.
Implements fixed-size chunking with configurable overlap.
"""
from dataclasses import dataclass
from typing import List
from ingestion import Document


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_index: int
    char_start: int
    char_end: int


class TextChunker:
    """
    Splits documents into overlapping fixed-size text chunks.

    Args:
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of characters of overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _chunk_text(self, text: str) -> List[tuple]:
        """
        Yield (char_start, char_end) pairs for each chunk.
        Tries to split on sentence/word boundaries when possible.
        """
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            # Try to end at a sentence boundary within the last 50 chars
            if end < text_len:
                boundary = text.rfind(". ", start, end)
                if boundary != -1 and boundary > start + self.chunk_size // 2:
                    end = boundary + 1  # include the period
                else:
                    # Try word boundary
                    space = text.rfind(" ", start, end)
                    if space != -1 and space > start + self.chunk_size // 2:
                        end = space
            chunks.append((start, end))
            # Next start with overlap
            next_start = end - self.overlap
            if next_start <= start:
                next_start = start + 1  # prevent infinite loop
            start = next_start
        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk a list of Documents into Chunk objects.
        Each document is chunked independently; chunk_index is global.
        """
        all_chunks = []
        global_idx = 0
        for doc in documents:
            if not doc.text.strip():
                continue
            spans = self._chunk_text(doc.text)
            for (char_start, char_end) in spans:
                chunk_text = doc.text[char_start:char_end].strip()
                if not chunk_text:
                    continue
                all_chunks.append(Chunk(
                    text=chunk_text,
                    source=doc.source,
                    page=doc.page,
                    chunk_index=global_idx,
                    char_start=char_start,
                    char_end=char_end,
                ))
                global_idx += 1
        return all_chunks


# Predefined configurations
CONFIG_A = TextChunker(chunk_size=256, overlap=32)
CONFIG_B = TextChunker(chunk_size=512, overlap=64)


if __name__ == "__main__":
    from ingestion import ingest_directory
    docs = ingest_directory("data")
    for name, cfg in [("CONFIG_A (256/32)", CONFIG_A), ("CONFIG_B (512/64)", CONFIG_B)]:
        chunks = cfg.chunk_documents(docs)
        avg_len = sum(len(c.text) for c in chunks) / max(len(chunks), 1)
        print(f"{name}: {len(chunks)} chunks, avg {avg_len:.1f} chars")
        if chunks:
            print("  Sample:", chunks[0].text[:120])
