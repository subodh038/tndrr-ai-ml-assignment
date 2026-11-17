"""
explanations.py - Technical Explanations for Information Retrieval and RAG Architecture.

This module provides formal documentation on:
1. Dense semantic embeddings vs. Sparse BM25 retrieval (Hybrid Search)
2. Cross-Encoder Rerankers: Mechanics and why they improve precision
3. Chunking trade-offs in Retrieval-Augmented Generation
4. Failure mode diagnostics and mitigation strategies
"""

BM25_AND_HYBRID_SEARCH = """
================================================================================
1. WHAT BM25 CONTRIBUTES TO SEMANTIC EMBEDDING SEARCH (HYBRID SEARCH)
================================================================================

Dense bi-encoder embeddings (e.g., all-MiniLM-L6-v2, text-embedding-ada-002) map 
arbitrary sequences into continuous low-dimensional vector spaces (e.g., 384 or 1536 dims). 
While exceptional at capturing broad semantic synonymy (e.g., matching "automobile" to "car"), 
dense embeddings exhibit severe blind spots:

1. Out-of-Vocabulary (OOV) and Exact Keyword Precision:
   - Dense embeddings frequently fail on specific entity names, code identifiers, 
     version numbers (e.g., "WMT 2014", "L=12, H=768"), chemical formulas, or rare acronyms.
   - Vector models smooth over rare tokens into generic latent neighborhood vectors.
   
2. Term Frequency Saturation & Document Length Normalization:
   - BM25 (Best Matching 25) calculates relevance using non-linear TF saturation and 
     strict length normalization:
     Score(D, Q) = sum_{t in Q} IDF(t) * [ TF(t,D) * (k1 + 1) ] / [ TF(t,D) + k1 * (1 - b + b * (|D| / avgdl)) ]
   - Unlike dense dot products, BM25 guarantees that if a rare query token appears multiple 
     times in a document, its contribution scales smoothly up to an asymptotic limit (governed by k1), 
     penalizing bloated passages (governed by b).

3. The Complementary Synergy of Hybrid Search:
   - Reciprocal Rank Fusion (RRF) or Convex Linear Combination:
     RRF_score(d) = sum_{m in {Dense, BM25}} 1 / (60 + rank_m(d))
   - Hybrid search achieves the best of both paradigms: BM25 acts as an exact-match hard filter 
     retaining crucial keywords, while Dense retrieval captures conceptual alignment, paraphrasing, 
     and semantic intent.
"""

RERANKER_MECHANICS = """
================================================================================
2. WHAT A RERANKER DOES AND WHY IT HELPS
================================================================================

In production RAG pipelines, retrieval is typically structured as a two-stage cascade:
  Stage 1: Fast Candidate Generation (Bi-Encoder Retrieval / Vector Search) -> Top 50-100 candidates
  Stage 2: Precision Re-ranking (Cross-Encoder / Late Interaction) -> Top 3-5 candidates

Why Bi-Encoders are Suboptimal at Ranking:
- Bi-encoders independently compute vector representations: v_q = E(q) and v_d = E(d).
- Similarity is computed via a cheap inner product: score = v_q . v_d.
- Crucially, tokens in query q NEVER interact directly with tokens in passage d. 
  Cross-attention between query and passage is completely absent!

How Cross-Encoder Rerankers Function:
- A Cross-Encoder feeds the concatenation of query and passage into the transformer simultaneously:
  Input = [CLS] + Query + [SEP] + Passage + [SEP]
- Every token of the query attends to every token of the candidate passage across all 
  transformer layers via full bidirectional self-attention:
  Attention(Q_all, K_all, V_all)
- The [CLS] token representation is passed through a classification head to output a calibrated 
  relevance score P(Relevant | Query, Passage).

Why Rerankers Drastically Improve RAG Precision:
1. Cross-Attention Expressivity: Captures nuanced word interactions, negation, prepositional logic, 
   and semantic dependencies that bi-encoder inner products compress and lose.
2. Suppression of False Positives: Dense retrieval frequently retrieves passages that share thematic 
   domain vocabulary but answer a completely different question. A cross-encoder discriminates 
   whether the passage actually answers the question.
3. Cost Efficiency: Running full cross-attention over millions of corpus documents is computationally 
   intractable (O(N) full forward passes). Re-ranking only the top 50 candidates from vector search 
   takes ~20-50ms while delivering near-optimal ranking accuracy.
"""

CHUNKING_TRADE_OFFS = """
================================================================================
3. CHUNKING SIZE AND OVERLAP TRADE-OFFS
================================================================================

- Configuration A (Small Chunks: size=256, overlap=32):
  + Advantages: Fine-grained embeddings, high semantic specificity, less noise injected into LLM context.
  - Disadvantages: Frequent boundary splitting of compound explanations, fragmented context, 
    loss of global document structure.

- Configuration B (Larger Chunks: size=512, overlap=64):
  + Advantages: Preserves complete semantic units, context surrounding key terms remains intact, 
    higher chance of containing the full answer + justification in a single chunk.
  - Disadvantages: Vector representations are diluted across multiple disparate ideas; potential 
    context window crowding in small-context LLMs.
"""

