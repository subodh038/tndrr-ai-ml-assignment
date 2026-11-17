"""
vector_store.py - FAISS-backed dense vector index for chunk retrieval.
"""
import os
from typing import Tuple
import numpy as np
import faiss


class VectorStore:
    """
    Wraps a FAISS IndexFlatL2 (exact nearest-neighbor search).

    For L2 on L2-normalized vectors, L2 distance = 2*(1 - cosine_sim),
    so this effectively does cosine retrieval when embeddings are normalized.

    Args:
        dim: Embedding dimensionality.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings: np.ndarray):
        """
        Add embeddings to the index.

        Args:
            embeddings: Float32 array of shape (N, dim).
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        print(f"[vector_store] Index now has {self.index.ntotal} vectors")

    def search(self, query_emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the k nearest neighbors of query_emb.

        Args:
            query_emb: Float32 array of shape (1, dim) or (dim,).
            k: Number of results to return.

        Returns:
            (distances, indices) — both shape (1, k).
            distances are L2 distances (smaller = more similar).
        """
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
        k_actual = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_emb, k_actual)
        return distances, indices

    def save(self, path: str):
        """Persist the FAISS index to disk."""
        faiss.write_index(self.index, path)
        print(f"[vector_store] Saved index ({self.index.ntotal} vectors) to {path}")

    def load(self, path: str):
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(path)
        self.dim = self.index.d
        print(f"[vector_store] Loaded index ({self.index.ntotal} vectors) from {path}")


if __name__ == "__main__":
    dim = 384
    store = VectorStore(dim)
    # Smoke test
    vecs = np.random.randn(10, dim).astype(np.float32)
    store.add(vecs)
    q = np.random.randn(1, dim).astype(np.float32)
    dists, idxs = store.search(q, k=3)
    print("Distances:", dists)
    print("Indices:", idxs)
