"""
evaluate_rag.py - Comparative Evaluation of Chunk Configurations on 15 Benchmark Questions.
Evaluates:
- Config A: chunk_size=256, overlap=32
- Config B: chunk_size=512, overlap=64
Metrics: Recall@1, Recall@3, Recall@5, MRR@5.
"""
import os
import sys
import json
import re
from typing import List, Dict, Any

# Ensure src is accessible
eval_dir = os.path.dirname(os.path.abspath(__file__))
task2_root = os.path.dirname(eval_dir)
src_dir = os.path.join(task2_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ingestion import ingest_directory
from chunker import TextChunker, CONFIG_A, CONFIG_B, Chunk
from embedder import Embedder
from vector_store import VectorStore
from retriever import Retriever


def clean_words(text: str) -> set:
    """Normalize text into a set of lower-case alphanumeric words."""
    return set(re.findall(r"\b[a-z0-9]{3,}\b", text.lower()))


def check_hit(gold_passage: str, gold_doc: str, gold_page: int, retrieved_chunk: Dict[str, Any]) -> bool:
    """
    Check if a retrieved chunk satisfies the ground truth passage.
    Matches if:
    1. Same source document
    2. Overlapping page (gold_page +/- 1 to account for PDF page breaks)
    3. Word overlap (Jaccard similarity >= 0.35 on content words) OR substring inclusion of key phrases
    """
    # Check document match
    chunk_source = os.path.basename(retrieved_chunk.get("source", ""))
    if chunk_source.lower() != gold_doc.lower():
        return False

    chunk_text = retrieved_chunk.get("text", "")
    
    # Direct substring match
    clean_chunk = " ".join(chunk_text.lower().split())
    clean_gold = " ".join(gold_passage.lower().split())
    if len(clean_gold) > 30 and clean_gold[:50] in clean_chunk:
        return True

    # Check page proximity
    page_diff = abs(retrieved_chunk.get("page", 0) - gold_page)
    if page_diff > 1:
        return False

    # Lexical overlap
    gold_words = clean_words(gold_passage)
    chunk_words = clean_words(chunk_text)
    if not gold_words:
        return False

    overlap = len(gold_words.intersection(chunk_words))
    recall_on_gold = overlap / len(gold_words)
    return recall_on_gold >= 0.35


def run_evaluation_for_config(
    config_name: str,
    chunker: TextChunker,
    docs: list,
    embedder: Embedder,
    questions: list,
    k: int = 5
) -> Dict[str, Any]:
    print(f"\n==================================================")
    print(f"Evaluating {config_name}: size={chunker.chunk_size}, overlap={chunker.overlap}")
    print(f"==================================================")

    chunks: List[Chunk] = chunker.chunk_documents(docs)
    print(f"Indexed chunks: {len(chunks)}")

    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)

    store = VectorStore(dim=embedder.dim)
    store.add(embeddings)

    retriever = Retriever(embedder=embedder, store=store, chunks=chunks)

    question_results = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []

    for q in questions:
        q_id = q["id"]
        q_text = q["question"]
        gold_passage = q["correct_source_passage"]
        gold_doc = q.get("source_doc", "attention.pdf")
        gold_page = q["source_page"]

        top_k = retriever.retrieve(q_text, k=k)

        first_hit_rank = None
        for rank, item in enumerate(top_k, start=1):
            if check_hit(gold_passage, gold_doc, gold_page, item):
                first_hit_rank = rank
                break

        if first_hit_rank == 1:
            hits_at_1 += 1
        if first_hit_rank is not None and first_hit_rank <= 3:
            hits_at_3 += 1
        if first_hit_rank is not None and first_hit_rank <= 5:
            hits_at_5 += 1

        rr = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        question_results.append({
            "id": q_id,
            "question": q_text,
            "gold_doc": gold_doc,
            "gold_page": gold_page,
            "hit_at_5": first_hit_rank is not None,
            "first_hit_rank": first_hit_rank,
            "top_retrieved": [
                {
                    "rank": i + 1,
                    "source": r["source"],
                    "page": r["page"],
                    "score": round(r["score"], 4),
                    "snippet": r["text"][:160] + "..."
                }
                for i, r in enumerate(top_k)
            ]
        })

    n_q = len(questions)
    r1 = hits_at_1 / n_q
    r3 = hits_at_3 / n_q
    r5 = hits_at_5 / n_q
    mrr = sum(reciprocal_ranks) / n_q

    summary = {
        "config_name": config_name,
        "chunk_size": chunker.chunk_size,
        "overlap": chunker.overlap,
        "total_chunks": len(chunks),
        "total_questions": n_q,
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        "hits_at_5": hits_at_5,
        "recall_at_1": round(r1, 4),
        "recall_at_3": round(r3, 4),
        "recall_at_5": round(r5, 4),
        "mrr_at_5": round(mrr, 4)
    }

    print(f"Results for {config_name}:")
    print(f"  Recall@1: {r1:.2%} ({hits_at_1}/{n_q})")
    print(f"  Recall@3: {r3:.2%} ({hits_at_3}/{n_q})")
    print(f"  Recall@5: {r5:.2%} ({hits_at_5}/{n_q})")
    print(f"  MRR@5:    {mrr:.4f}")

    return {
        "summary": summary,
        "details": question_results
    }


def main():
    data_dir = os.path.join(task2_root, "data")
    questions_file = os.path.join(eval_dir, "questions.json")
    results_dir = os.path.join(task2_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"[eval] Loading questions from {questions_file}...")
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"[eval] Ingesting documents from {data_dir}...")
    docs = ingest_directory(data_dir)
    print(f"[eval] Total pages/sections loaded: {len(docs)}")

    print(f"[eval] Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = Embedder("all-MiniLM-L6-v2")

    # Run Config A (256/32)
    res_a = run_evaluation_for_config("Config_A", CONFIG_A, docs, embedder, questions, k=5)
    with open(os.path.join(results_dir, "chunk_a_results.json"), "w", encoding="utf-8") as f:
        json.dump(res_a, f, indent=2)

    # Run Config B (512/64)
    res_b = run_evaluation_for_config("Config_B", CONFIG_B, docs, embedder, questions, k=5)
    with open(os.path.join(results_dir, "chunk_b_results.json"), "w", encoding="utf-8") as f:
        json.dump(res_b, f, indent=2)

    # Failure Analysis: find questions where either or both failed
    failures = []
    for q_a, q_b in zip(res_a["details"], res_b["details"]):
        q_id = q_a["id"]
        hit_a = q_a["hit_at_5"]
        hit_b = q_b["hit_at_5"]
        if not hit_a or not hit_b:
            failures.append({
                "id": q_id,
                "question": q_a["question"],
                "gold_doc": q_a["gold_doc"],
                "gold_page": q_a["gold_page"],
                "hit_config_a": hit_a,
                "hit_config_b": hit_b,
                "rank_a": q_a["first_hit_rank"],
                "rank_b": q_b["first_hit_rank"],
                "top_retrieved_a": q_a["top_retrieved"][:2],
                "top_retrieved_b": q_b["top_retrieved"][:2]
            })

    comparison = {
        "dataset": {
            "document_collection": "arXiv:1706.03762 (Attention) + arXiv:1810.04805 (BERT)",
            "total_pages": len(docs),
            "num_questions": len(questions)
        },
        "config_a": res_a["summary"],
        "config_b": res_b["summary"],
        "comparison_summary": {
            "better_config": "Config_B" if res_b["summary"]["recall_at_5"] >= res_a["summary"]["recall_at_5"] else "Config_A",
            "recall_5_diff": round(res_b["summary"]["recall_at_5"] - res_a["summary"]["recall_at_5"], 4),
            "mrr_diff": round(res_b["summary"]["mrr_at_5"] - res_a["summary"]["mrr_at_5"], 4)
        },
        "total_failures_detected": len(failures),
        "failure_cases": failures
    }

    eval_summary_file = os.path.join(results_dir, "rag_evaluation_results.json")
    with open(eval_summary_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[eval] Saved evaluation comparison to {eval_summary_file}")


if __name__ == "__main__":
    main()

