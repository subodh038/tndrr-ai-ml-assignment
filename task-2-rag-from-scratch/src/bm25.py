"""
bm25.py - BM25 Okapi implementation from scratch (no external BM25 libraries).

BM25 is a classic sparse retrieval function that ranks documents by term
frequency (TF) and inverse document frequency (IDF), with length normalization.
"""
import math
import re
from collections import Counter
from typing import List


def _tokenize(text: str) -> List[str]:
    """Simple whitespace+punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """
    BM25 Okapi retrieval model.

    Score formula:
        sum_t [ IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1 - b + b*|d|/avgdl)) ]
    where IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

    Args:
        corpus: List of document strings.
        k1: TF saturation parameter (default 1.5). Higher = more TF weight.
        b: Length normalization factor (default 0.75). 0 = no normalization.
    """

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)

        # Tokenize all documents
        self.tokenized = [_tokenize(doc) for doc in corpus]
        self.doc_len = [len(toks) for toks in self.tokenized]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)

        # Term frequencies per document
        self.tf = [Counter(toks) for toks in self.tokenized]

        # Document frequency: how many docs contain each term
        self.df: dict = {}
        for tok_list in self.tokenized:
            for term in set(tok_list):
                self.df[term] = self.df.get(term, 0) + 1

        # Precompute IDF
        self.idf: dict = {}
        for term, df_t in self.df.items():
            self.idf[term] = math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1.0)

    def score(self, query: str, doc_index: int) -> float:
        """
        Compute BM25 score for a single document.

        Args:
            query: Query string.
            doc_index: Index of the document in the corpus.

        Returns:
            BM25 score (higher = more relevant).
        """
        query_terms = _tokenize(query)
        doc_tf = self.tf[doc_index]
        dl = self.doc_len[doc_index]
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            tf_td = doc_tf.get(term, 0)
            idf_t = self.idf[term]
            numerator = tf_td * (self.k1 + 1)
            denominator = tf_td + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf_t * numerator / denominator
        return score

    def search(self, query: str, k: int = 5) -> List[int]:
        """
        Return the indices of the top-k most relevant documents.

        Args:
            query: Query string.
            k: Number of results.

        Returns:
            List of document indices sorted by descending BM25 score.
        """
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:k]]

    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Return (index, score) tuples for top-k docs."""
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


if __name__ == "__main__":
    corpus = [
        "The transformer architecture uses self-attention mechanisms.",
        "BERT is a bidirectional encoder representation from transformers.",
        "GPT uses autoregressive language modeling with transformers.",
        "Retrieval augmented generation combines retrieval with language models.",
        "Fine-tuning adapts pretrained models to downstream tasks.",
    ]
    bm25 = BM25(corpus)
    query = "transformer attention"
    results = bm25.search_with_scores(query, k=3)
    for idx, sc in results:
        print(f"[{sc:.3f}] {corpus[idx][:80]}")
