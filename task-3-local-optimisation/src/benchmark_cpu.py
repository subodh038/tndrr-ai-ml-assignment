"""
CPU Benchmark for flan-t5-small (FP32 and INT8 dynamic quantization).
Measures: load time, model memory, tokens/sec, latency per prompt.
"""
import time
import json
import psutil
import os

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def measure_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_benchmark(model_name: str, prompts: list, quantize: bool = False) -> dict:
    """
    Run a full benchmark of model_name on the given prompts.

    Args:
        model_name: HuggingFace model identifier (e.g. 'google/flan-t5-small').
        prompts: List of prompt dicts with keys: id, category, prompt.
        quantize: If True, apply dynamic INT8 quantization (torch.qint8) to Linear layers.

    Returns:
        dict with per-prompt results and aggregate summary.
    """
    results = {
        "model": model_name,
        "quantized": quantize,
        "dtype": "INT8 (dynamic)" if quantize else "FP32",
        "prompts": [],
        "summary": {},
    }

    baseline_mem = measure_memory_mb()
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if quantize:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    model.eval()
    load_time = time.time() - load_start
    post_load_mem = measure_memory_mb()
    model_mem_mb = post_load_mem - baseline_mem

    total_tokens = 0
    total_time = 0.0

    for p in prompts:
        inputs = tokenizer(
            p["prompt"], return_tensors="pt", truncation=True, max_length=512
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)
        t1 = time.perf_counter()

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        n_tokens = outputs[0].shape[0]
        latency = t1 - t0
        tps = n_tokens / latency
        total_tokens += n_tokens
        total_time += latency

        results["prompts"].append(
            {
                "id": p["id"],
                "category": p["category"],
                "prompt": p["prompt"],
                "output": generated_text,
                "tokens": int(n_tokens),
                "latency_s": round(latency, 3),
                "tokens_per_sec": round(tps, 2),
            }
        )
        print(
            f"  Prompt {p['id']:2d} [{p['category']:14s}] "
            f"→ {n_tokens} tokens in {latency:.3f}s ({tps:.1f} tok/s): {generated_text[:60]!r}"
        )

    results["summary"] = {
        "load_time_s": round(load_time, 3),
        "model_memory_mb": round(model_mem_mb, 1),
        "avg_tokens_per_sec": round(total_tokens / total_time, 2),
        "avg_latency_s": round(total_time / len(prompts), 3),
        "total_tokens_generated": int(total_tokens),
    }
    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_path = os.path.join(base_dir, "prompts", "benchmark_prompts.json")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    with open(prompts_path) as f:
        prompts = json.load(f)

    # ---- FP32 run ----
    print("\n=== FP32 benchmark ===")
    r_fp32 = run_benchmark("google/flan-t5-small", prompts, quantize=False)
    fp32_path = os.path.join(results_dir, "cpu_fp32_outputs.json")
    with open(fp32_path, "w") as f:
        json.dump(r_fp32, f, indent=2)
    print("FP32 summary:", r_fp32["summary"])

    # ---- INT8 dynamic quantization run ----
    print("\n=== INT8 (dynamic) benchmark ===")
    r_int8 = run_benchmark("google/flan-t5-small", prompts, quantize=True)
    int8_path = os.path.join(results_dir, "cpu_int8_outputs.json")
    with open(int8_path, "w") as f:
        json.dump(r_int8, f, indent=2)
    print("INT8 summary:", r_int8["summary"])

    print("\nDone. Results saved to", results_dir)
