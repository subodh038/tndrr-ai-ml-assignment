"""
embedder.py - Sentence embedding utilities using sentence-transformers.
"""
from typing import List
import numpy as np


class Embedder:
    """
    Wraps a SentenceTransformer model to embed text passages.

    Args:
        model_name: HuggingFace model name (default: all-MiniLM-L6-v2).
                    This model produces 384-dim embeddings and is fast on CPU.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        print(f"[embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[embedder] Embedding dimension: {self.dim}")

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts per forward pass.

        Returns:
            np.ndarray of shape (len(texts), dim), dtype float32.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
        )
        return embeddings.astype(np.float32)

    def save(self, embeddings: np.ndarray, path: str):
        """Save embeddings array to a .npy file."""
        np.save(path, embeddings)
        print(f"[embedder] Saved {embeddings.shape} embeddings to {path}")

    def load(self, path: str) -> np.ndarray:
        """Load embeddings array from a .npy file."""
        emb = np.load(path)
        print(f"[embedder] Loaded {emb.shape} embeddings from {path}")
        return emb


if __name__ == "__main__":
    e = Embedder()
    sample = ["Hello world", "Transformer attention mechanism", "BERT pre-training"]
    vecs = e.embed(sample)
    print("Shape:", vecs.shape)
    print("Norms:", np.linalg.norm(vecs, axis=1))
