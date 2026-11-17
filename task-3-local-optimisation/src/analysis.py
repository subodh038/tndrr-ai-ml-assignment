"""
analysis.py — Conceptual Analysis Module for Local LLM Optimisation
====================================================================

This module provides:
  1. KV cache memory estimation formula and explanation.
  2. Quantisation memory savings calculator.
  3. 20-user concurrent serving plan on a 24 GB GPU.
  4. Theoretical vs measured comparison (populated after benchmark run).

Key concepts
------------

KV Cache
~~~~~~~~
During transformer inference the model computes Key (K) and Value (V) tensors
for every attention head in every layer.  Rather than recompute them for the
tokens already processed, the runtime caches them in memory.

Memory consumed per sequence:
  2  (K + V)
  × n_layers
  × n_kv_heads × head_dim   (= d_model when using Multi-Head Attention)
  × seq_len
  × dtype_bytes              (2 for fp16/bf16, 4 for fp32)

This grows linearly with sequence length and batch size, making it the dominant
memory consumer at long contexts.

Quantisation
~~~~~~~~~~~~
Quantisation reduces the bit-width used to store model weights (and optionally
activations).  Common schemes:

  • FP32 → INT8  (dynamic):   4× memory reduction, slight accuracy loss.
  • FP16 → INT4  (Q4_K_M):   4× reduction on top of fp16, used in GGUF/llama.cpp.

Dynamic quantisation (used here) quantises only weights; activations remain
float32, so accuracy degradation is minimal while memory and compute savings
are real on CPU (via vectorised integer GEMM).

Quality / Performance Trade-offs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  • INT8 dynamic: ~20-40% memory reduction, 10-30% faster on CPU, negligible
    quality loss on seq2seq tasks (T5, BART).
  • INT4 (GPTQ/GGUF): ~75% memory reduction vs FP16, 1-2 perplexity points
    increase on LLaMA-class models—acceptable for most applications.
  • Speculative decoding: draft model proposes tokens; large model verifies
    in parallel—2-4× throughput with zero quality loss.
  • Flash Attention: memory-efficient attention kernel, O(1) extra memory in
    sequence length; no quality change.
"""

from __future__ import annotations
import json
import os


# ---------------------------------------------------------------------------
# KV cache memory
# ---------------------------------------------------------------------------

def kv_cache_memory(
    n_layers: int,
    d_model: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
) -> dict:
    """
    Estimate KV cache memory for a *standard Multi-Head Attention* model.

    Formula:
        bytes = 2 × n_layers × d_model × seq_len × batch_size × dtype_bytes

    For GQA/MQA models replace d_model with (n_kv_heads × head_dim).

    Args:
        n_layers:    Number of transformer layers.
        d_model:     Full model hidden size (= n_heads × head_dim for MHA).
        seq_len:     Maximum sequence length (prompt + generation).
        batch_size:  Number of concurrent sequences.
        dtype_bytes: Bytes per element (2 = fp16/bf16, 4 = fp32).

    Returns:
        dict with bytes, KB, MB, GB values.

    Example — Llama-3-8B (MHA, 32 layers, d_model=4096, ctx=2048, batch=1, fp16):
        kv_cache_memory(32, 4096, 2048, 1, 2)  → ~536 MB
    """
    total_bytes = 2 * n_layers * d_model * seq_len * batch_size * dtype_bytes
    return {
        "bytes": total_bytes,
        "kb": total_bytes / 1024,
        "mb": round(total_bytes / 1024**2, 2),
        "gb": round(total_bytes / 1024**3, 4),
        "params": {
            "n_layers": n_layers,
            "d_model": d_model,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "dtype_bytes": dtype_bytes,
        },
    }


def kv_cache_memory_gqa(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
) -> dict:
    """
    KV cache for Grouped Query Attention (GQA) — e.g. Llama-3 / Mistral.

    Formula:
        bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch_size × dtype_bytes

    For Llama-3-8B: n_kv_heads=8, head_dim=128 (not 4096 full d_model).
    """
    total_bytes = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    return {
        "bytes": total_bytes,
        "kb": total_bytes / 1024,
        "mb": round(total_bytes / 1024**2, 2),
        "gb": round(total_bytes / 1024**3, 4),
    }


# ---------------------------------------------------------------------------
# Quantisation memory savings
# ---------------------------------------------------------------------------

def quantization_memory_savings(
    n_params: int,
    from_bits: int = 16,
    to_bits: int = 4,
) -> dict:
    """
    Calculate memory reduction from quantisation.

    Args:
        n_params:  Number of model parameters (e.g. 7_000_000_000 for 7B).
        from_bits: Original precision (16 = fp16, 32 = fp32).
        to_bits:   Target precision (8 = int8, 4 = int4).

    Returns:
        dict with original_gb, quantized_gb, reduction_factor, saved_gb.

    Example — 7B model fp16→int4:
        quantization_memory_savings(7e9, 16, 4)
        → {'original_gb': 14.0, 'quantized_gb': 3.5, 'reduction_factor': 4.0}
    """
    original_bytes = n_params * (from_bits / 8)
    quantized_bytes = n_params * (to_bits / 8)
    return {
        "n_params": n_params,
        "from_bits": from_bits,
        "to_bits": to_bits,
        "original_gb": round(original_bytes / 1e9, 3),
        "quantized_gb": round(quantized_bytes / 1e9, 3),
        "saved_gb": round((original_bytes - quantized_bytes) / 1e9, 3),
        "reduction_factor": round(original_bytes / quantized_bytes, 2),
    }


# ---------------------------------------------------------------------------
# 20-user concurrent serving plan
# ---------------------------------------------------------------------------

def serving_plan_20_users() -> dict:
    """
    Design a 20-user concurrent serving plan on a single 24 GB GPU.

    Model choice
    ------------
    Llama-3-8B quantised to Q4_K_M (≈4-bit average):
      • Weight memory ≈ 8B × 0.5 bytes = 4 GB (+ ~0.5 GB overhead) ≈ 4.5 GB

    Available VRAM for KV cache
    ---------------------------
      24 GB total − 4.5 GB model = 19.5 GB ≈ 19 GB (conservative)

    KV cache per token (Llama-3-8B GQA, fp16)
    ------------------------------------------
      2 (K+V) × 8 (kv_heads) × 128 (head_dim) × 32 (layers) × 2 (fp16 bytes)
      = 131 072 bytes ≈ 128 KB / token

    KV cache per 2048-token sequence
    ---------------------------------
      131 072 × 2 048 = 268 435 456 bytes ≈ 256 MB / sequence

    Max concurrent sequences
    -------------------------
      19 GB / 256 MB ≈ 76 sequences
      → comfortably supports 20 users with 2048-token contexts

    Batching & queue design
    -----------------------
      • Continuous batching (vLLM-style): new requests fill slots freed by
        completed sequences without waiting for the whole batch to finish.
      • Rate limiting: cap at 20 active + 20 queued; return 503 if queue full.
      • Per-user SLA: p50 TTFT < 500 ms, p99 < 2 s.
    """
    n_kv_heads = 8
    head_dim = 128
    n_layers = 32
    dtype_bytes = 2  # fp16

    kv_per_token_bytes = 2 * n_kv_heads * head_dim * n_layers * dtype_bytes  # 131072

    model_vram_gb = 4.5          # Q4_K_M 8B + overhead
    total_vram_gb = 24.0
    kv_budget_gb  = total_vram_gb - model_vram_gb   # 19.5 GB
    context_length = 2048

    kv_per_seq_bytes = kv_per_token_bytes * context_length
    max_sequences = int((kv_budget_gb * 1024**3) / kv_per_seq_bytes)

    plan = {
        "model": "Llama-3-8B Q4_K_M",
        "gpu_vram_gb": total_vram_gb,
        "model_vram_gb": model_vram_gb,
        "kv_cache_budget_gb": round(kv_budget_gb, 1),
        "kv_per_token_bytes": kv_per_token_bytes,
        "kv_per_token_kb": round(kv_per_token_bytes / 1024, 1),
        "context_length_tokens": context_length,
        "kv_per_sequence_mb": round(kv_per_seq_bytes / 1024**2, 1),
        "max_concurrent_sequences_theoretical": max_sequences,
        "target_concurrent_users": 20,
        "headroom_factor": round(max_sequences / 20, 1),
        "batching_strategy": "continuous batching (vLLM PagedAttention)",
        "queue": {
            "max_active": 20,
            "max_queued": 20,
            "overflow_response": "HTTP 503 Service Unavailable",
        },
        "sla": {
            "p50_ttft_ms": 500,
            "p99_ttft_ms": 2000,
        },
        "optimisations": [
            "Flash Attention 2 for memory-efficient attention",
            "Speculative decoding with Llama-3.2-1B draft model (est. 2-3× throughput)",
            "KV cache prefix sharing for common system prompts",
            "Request batching with dynamic padding",
        ],
    }
    return plan


# ---------------------------------------------------------------------------
# Theoretical vs measured comparison
# ---------------------------------------------------------------------------

def theoretical_vs_measured_comparison(results_dir: str | None = None) -> dict:
    """
    Compare theoretical estimates with actual benchmark measurements.

    Theoretical estimates (flan-t5-small, 77M params):
      • FP32 weight memory: 77M × 4 bytes = 308 MB
      • INT8 weight memory: 77M × 1 byte  =  77 MB  (dynamic INT8 Linear only)
      • Memory reduction: 4× (weights only; actual will differ due to activations/runtime)

    Measured numbers are loaded from results/cpu_fp32_outputs.json and
    results/cpu_int8_outputs.json when available.

    Returns:
        dict with 'theoretical' and 'measured' sub-dicts, clearly labelled.
    """
    n_params = 77_000_000  # flan-t5-small

    theoretical = {
        "source": "THEORETICAL",
        "model": "google/flan-t5-small (77M params)",
        "fp32_weight_memory_mb": round(n_params * 4 / 1024**2, 1),
        "int8_weight_memory_mb": round(n_params * 1 / 1024**2, 1),
        "expected_memory_reduction_factor": 4.0,
        "note": (
            "Dynamic INT8 quantises only Linear layer weights; embedding and "
            "layer-norm weights stay FP32, so actual reduction is less than 4×."
        ),
    }

    measured: dict = {"source": "MEASURED on this machine (CPU-only)"}

    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
        )

    fp32_path = os.path.join(results_dir, "cpu_fp32_outputs.json")
    int8_path = os.path.join(results_dir, "cpu_int8_outputs.json")

    if os.path.exists(fp32_path) and os.path.exists(int8_path):
        with open(fp32_path) as f:
            fp32 = json.load(f)
        with open(int8_path) as f:
            int8 = json.load(f)

        s32 = fp32["summary"]
        s8  = int8["summary"]
        measured.update({
            "fp32": s32,
            "int8": s8,
            "actual_memory_reduction_factor": round(
                s32["model_memory_mb"] / s8["model_memory_mb"], 2
            ) if s8["model_memory_mb"] > 0 else None,
            "actual_tps_speedup": round(
                s8["avg_tokens_per_sec"] / s32["avg_tokens_per_sec"], 2
            ) if s32["avg_tokens_per_sec"] > 0 else None,
        })
    else:
        measured["status"] = "NOT YET AVAILABLE — run benchmark_cpu.py first"

    return {"theoretical": theoretical, "measured": measured}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== KV Cache Memory (Llama-3-8B, GQA, ctx=2048, batch=1, fp16) ===")
    kv = kv_cache_memory_gqa(32, 8, 128, 2048, 1, 2)
    print(f"  {kv['mb']:.1f} MB per sequence")

    print("\n=== Quantisation Savings (7B fp16 → int4) ===")
    qs = quantization_memory_savings(7_000_000_000, 16, 4)
    print(f"  {qs['original_gb']} GB → {qs['quantized_gb']} GB  ({qs['reduction_factor']}× reduction)")

    print("\n=== 20-User Serving Plan ===")
    plan = serving_plan_20_users()
    for k, v in plan.items():
        print(f"  {k}: {v}")

    print("\n=== Theoretical vs Measured ===")
    comparison = theoretical_vs_measured_comparison()
    print(json.dumps(comparison, indent=2))
