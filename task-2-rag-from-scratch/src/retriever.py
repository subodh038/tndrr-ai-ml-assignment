"""
retriever.py - Query encoder + FAISS lookup + result formatting.
"""
from typing import List
import numpy as np
from embedder import Embedder
from vector_store import VectorStore
from chunker import Chunk


class Retriever:
    """
    Encodes a query string, looks up top-k chunks in the vector store,
    and formats the results.

    Args:
        embedder: Embedder instance for encoding queries.
        store: VectorStore containing indexed chunk embeddings.
        chunks: List of Chunk objects in the same order as the index.
    """

    def __init__(self, embedder: Embedder, store: VectorStore, chunks: List[Chunk]):
        self.embedder = embedder
        self.store = store
        self.chunks = chunks

    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: Natural language question or query string.
            k: Number of results to return.

        Returns:
            List of dicts: [{text, source, page, score, chunk_index}, ...]
            Sorted by relevance (best first). Score is cosine similarity
            (1 - L2_distance/2 for normalized vectors).
        """
        query_emb = self.embedder.embed([query])  # (1, dim)
        distances, indices = self.store.search(query_emb, k=k)
        # distances are L2 on normalized vectors; convert to cosine similarity
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            # For L2-normalized vectors: cosine_sim = 1 - dist/2
            cosine_sim = float(1.0 - dist / 2.0)
            results.append({
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "score": cosine_sim,
                "chunk_index": chunk.chunk_index,
            })
        return results

    def build_context(self, results: List[dict], max_tokens: int = 1000) -> str:
        """
        Concatenate retrieved passages into a context string for generation.
        Respects an approximate token budget (1 token ≈ 4 chars).

        Args:
            results: Output of retrieve().
            max_tokens: Approximate token limit for the context.

        Returns:
            Formatted context string with source citations.
        """
        max_chars = max_tokens * 4
        parts = []
        used = 0
        for i, r in enumerate(results):
            header = f"[Source {i+1}: {r['source']}, page {r['page']}]\n"
            snippet = r["text"]
            block = header + snippet + "\n\n"
            if used + len(block) > max_chars:
                remaining = max_chars - used - len(header) - 2
                if remaining > 50:
                    block = header + snippet[:remaining] + "...\n\n"
                else:
                    break
            parts.append(block)
            used += len(block)
        return "".join(parts).strip()


if __name__ == "__main__":
    from ingestion import ingest_directory
    from chunker import CONFIG_A
    import os, sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    docs = ingest_directory(data_dir)
    chunks = CONFIG_A.chunk_documents(docs)
    texts = [c.text for c in chunks]

    emb = Embedder()
    vectors = emb.embed(texts)

    store = VectorStore(dim=emb.dim)
    store.add(vectors)

    retriever = Retriever(emb, store, chunks)
    results = retriever.retrieve("What is the attention mechanism?", k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['source']} p{r['page']}: {r['text'][:100]}")
