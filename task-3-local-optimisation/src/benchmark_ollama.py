"""
benchmark_ollama.py — Ollama Local Model Benchmark Suite
=========================================================
Benchmarks two quantization variants of Llama 3.2 1B:
  - llama3.2:1b-instruct-q4_K_M
  - llama3.2:1b-instruct-q8_0

Measures:
  - Cold/Warm load memory via Ollama's /api/ps (model size & size_vram)
  - System RAM usage before and after loading
  - Per-prompt latency, token count, and generation tokens/sec (eval_count / eval_duration)
  - Aggregate generation tokens/sec (total generated tokens / total evaluation time)
  - Identical generation parameters (num_predict, temperature, num_ctx)

Saves:
  - results/ollama_<model_name>.json (per-model prompt outputs and metrics)
  - results/ollama_comparison.json (combined side-by-side JSON)
  - results/ollama_comparison.csv (readable comparison CSV)
"""

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

TARGET_MODELS = [
    "llama3.2:1b-instruct-q4_K_M",
    "llama3.2:1b-instruct-q8_0",
]

# Strict generation parameters applied identically to all models
GENERATION_OPTIONS = {
    "num_predict": 128,  # Max new tokens to generate
    "temperature": 0.7,  # Sampling temperature
    "num_ctx": 2048,     # Context window length
}

TIMEOUT_SECONDS = 180


# ── Health & Diagnostic Helpers ───────────────────────────────────────────────

def check_ollama_running(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Check if the Ollama daemon is reachable."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_available_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    """Retrieve list of pulled model names from Ollama."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        names = []
        for m in models:
            name = m.get("name", "")
            names.append(name)
            # Also register name without tag if it has ':latest'
            if ":latest" in name:
                names.append(name.replace(":latest", ""))
        return names
    except Exception as e:
        print(f"[ERROR] Failed to query /api/tags: {e}", file=sys.stderr)
        return []


def query_running_model_memory(
    model_name: str, base_url: str = OLLAMA_BASE_URL
) -> Dict[str, Any]:
    """
    Query Ollama's /api/ps endpoint to capture model size and VRAM allocation.

    Returns:
        Dict with keys:
          - model_size_bytes: int or None
          - model_size_mb: float or None
          - vram_bytes: int or None
          - vram_mb: float or None
          - vram_status: str ("Allocated", "Unavailable", etc.)
          - quantization: str or None
    """
    result = {
        "model_size_bytes": None,
        "model_size_mb": None,
        "vram_bytes": None,
        "vram_mb": None,
        "vram_status": "Unavailable",
        "quantization": None,
        "details": {},
    }

    try:
        resp = requests.get(f"{base_url}/api/ps", timeout=5)
        resp.raise_for_status()
        running_models = resp.json().get("models", [])

        # Match model by exact name or prefix
        target_info = None
        for m in running_models:
            m_name = m.get("name", "")
            m_model = m.get("model", "")
            if model_name in (m_name, m_model) or model_name.startswith(m_name):
                target_info = m
                break

        if target_info:
            size_b = target_info.get("size")
            vram_b = target_info.get("size_vram")
            details = target_info.get("details", {})

            if size_b is not None:
                result["model_size_bytes"] = size_b
                result["model_size_mb"] = round(size_b / (1024 * 1024), 2)

            if vram_b is not None and vram_b > 0:
                result["vram_bytes"] = vram_b
                result["vram_mb"] = round(vram_b / (1024 * 1024), 2)
                result["vram_status"] = f"{result['vram_mb']:.1f} MB (GPU Allocated)"
            elif vram_b == 0:
                result["vram_bytes"] = 0
                result["vram_mb"] = 0.0
                result["vram_status"] = "0.0 MB (CPU Mode / No VRAM offloaded)"
            else:
                result["vram_status"] = "Unavailable"

            result["quantization"] = details.get("quantization_level")
            result["details"] = details
        else:
            result["vram_status"] = "Unavailable (Model not active in /api/ps)"

    except Exception as e:
        result["vram_status"] = f"Unavailable (Query error: {e})"

    return result


def get_system_ram_mb() -> float:
    """Return total system RAM currently in use (MB)."""
    return round(psutil.virtual_memory().used / (1024 * 1024), 2)


# ── Benchmark Execution Engine ────────────────────────────────────────────────

def run_warmup_request(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    """
    Perform a single warm-up generation request to ensure the model weights
    and KV cache structures are loaded into server memory before timing.
    """
    print(f"  [Warm-up] Sending initial request to load {model_name} into memory...")
    payload = {
        "model": model_name,
        "prompt": "Respond with the single word: READY.",
        "stream": False,
        "options": GENERATION_OPTIONS,
    }
    try:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{base_url}/api/generate", json=payload, timeout=TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        elapsed = time.perf_counter() - t0
        data = resp.json()
        print(
            f"  [Warm-up] Completed in {elapsed:.2f}s "
            f"(eval_count={data.get('eval_count', 0)}, load_duration={data.get('load_duration', 0)/1e9:.2f}s)"
        )
        return True
    except Exception as e:
        print(f"  [ERROR] Warm-up request failed for {model_name}: {e}", file=sys.stderr)
        return False


def benchmark_single_model(
    model_name: str,
    prompts: List[Dict[str, Any]],
    base_url: str = OLLAMA_BASE_URL,
) -> Dict[str, Any]:
    """
    Benchmark an Ollama model against the 10 prompts.

    Records:
      - Host RAM before and after load
      - /api/ps memory metrics (model size & size_vram)
      - Per-prompt latency, tokens, tokens/sec
      - Aggregate generation tokens/sec = total_eval_count / total_eval_duration_s
    """
    print(f"\n{'='*72}")
    print(f"Benchmarking Model: {model_name}")
    print(f"{'='*72}")

    ram_before_mb = get_system_ram_mb()

    # 1. Warm-up request
    warmup_ok = run_warmup_request(model_name, base_url)
    if not warmup_ok:
        raise RuntimeError(f"Warm-up failed for model '{model_name}'. Aborting run.")

    # Small settle pause to let server update /api/ps
    time.sleep(1.0)

    # 2. Capture memory state via /api/ps and system RAM
    ram_after_load_mb = get_system_ram_mb()
    ps_memory = query_running_model_memory(model_name, base_url)
    ram_delta_mb = round(ram_after_load_mb - ram_before_mb, 2)

    print(f"  [Memory State]")
    print(f"    - Model Size (Disk/RAM): {ps_memory.get('model_size_mb', 'N/A')} MB")
    print(f"    - VRAM Allocation      : {ps_memory.get('vram_status')}")
    print(f"    - Quantization Level   : {ps_memory.get('quantization', 'N/A')}")
    print(f"    - System RAM Before    : {ram_before_mb:.1f} MB")
    print(f"    - System RAM After Load: {ram_after_load_mb:.1f} MB (Delta: {ram_delta_mb:+.1f} MB)")
    print(f"    - Python Process RSS   : {psutil.Process(os.getpid()).memory_info().rss / (1024*1024):.1f} MB (Client runner, not model)")

    # 3. Timed benchmark loop over 10 prompts
    prompt_records = []
    total_eval_tokens = 0
    total_eval_duration_ns = 0
    total_wall_clock_s = 0.0

    print(f"\n  [Inference on {len(prompts)} Benchmark Prompts]")
    for p in prompts:
        p_id = p["id"]
        cat = p["category"]
        prompt_text = p["prompt"]

        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "stream": False,
            "options": GENERATION_OPTIONS,
        }

        t_start = time.perf_counter()
        try:
            resp = requests.post(
                f"{base_url}/api/generate", json=payload, timeout=TIMEOUT_SECONDS
            )
            resp.raise_for_status()
            t_end = time.perf_counter()
            data = resp.json()
        except Exception as e:
            print(f"    [FAIL] Prompt {p_id} failed: {e}", file=sys.stderr)
            prompt_records.append({
                "id": p_id,
                "category": cat,
                "prompt": prompt_text,
                "error": str(e),
                "output": "",
                "eval_count": 0,
                "eval_duration_s": 0.0,
                "latency_s": round(time.perf_counter() - t_start, 3),
                "generation_tokens_per_sec": 0.0,
            })
            continue

        latency_s = t_end - t_start
        generated_text = data.get("response", "").strip()

        # Ollama evaluation metrics
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        eval_duration_s = eval_duration_ns / 1e9 if eval_duration_ns else 0.0

        # Prompt eval metrics (prompt ingestion)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_duration_s = (data.get("prompt_eval_duration", 0) or 0) / 1e9

        # Generation tokens per second (eval_count / eval_duration)
        if eval_duration_s > 0:
            prompt_tps = eval_count / eval_duration_s
        else:
            prompt_tps = 0.0

        total_eval_tokens += eval_count
        total_eval_duration_ns += eval_duration_ns
        total_wall_clock_s += latency_s

        record = {
            "id": p_id,
            "category": cat,
            "prompt": prompt_text,
            "output": generated_text,
            "output_tokens": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration_s": round(prompt_eval_duration_s, 4),
            "eval_duration_s": round(eval_duration_s, 4),
            "latency_s": round(latency_s, 3),
            "generation_tokens_per_sec": round(prompt_tps, 2),
        }
        prompt_records.append(record)

        print(
            f"    Prompt {p_id:02d} [{cat:14s}] → {eval_count:3d} tokens in "
            f"{latency_s:.3f}s (gen: {prompt_tps:6.2f} tok/s, eval_dur: {eval_duration_s:.3f}s)"
        )

    # 4. Compute aggregate metrics
    total_eval_duration_s = total_eval_duration_ns / 1e9
    if total_eval_duration_s > 0:
        aggregate_generation_tps = total_eval_tokens / total_eval_duration_s
    else:
        aggregate_generation_tps = 0.0

    avg_latency_s = total_wall_clock_s / len(prompts) if prompts else 0.0

    summary = {
        "model": model_name,
        "quantization": ps_memory.get("quantization") or "Unknown",
        "model_size_mb": ps_memory.get("model_size_mb"),
        "vram_mb": ps_memory.get("vram_mb"),
        "vram_status": ps_memory.get("vram_status"),
        "system_ram_before_mb": ram_before_mb,
        "system_ram_after_load_mb": ram_after_load_mb,
        "system_ram_delta_mb": ram_delta_mb,
        "generation_options": GENERATION_OPTIONS,
        "num_prompts": len(prompts),
        "total_output_tokens": total_eval_tokens,
        "total_eval_duration_s": round(total_eval_duration_s, 4),
        "total_wall_clock_s": round(total_wall_clock_s, 3),
        "avg_latency_s": round(avg_latency_s, 3),
        "aggregate_generation_tokens_per_sec": round(aggregate_generation_tps, 2),
    }

    return {
        "model_info": ps_memory,
        "summary": summary,
        "prompts": prompt_records,
    }


# ── Comparison Reporting (JSON, CSV, Console Table) ───────────────────────────

def build_comparison_structures(
    results_map: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate structured comparison dictionaries for JSON and CSV exports."""
    models = list(results_map.keys())

    comparison_json = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_settings": GENERATION_OPTIONS,
        "models": {},
        "comparison_metrics": {},
    }

    # CSV rows: List of dicts with metric and column values
    csv_rows = []

    for m in models:
        s = results_map[m]["summary"]
        comparison_json["models"][m] = s

    # Define comparison rows
    metrics_spec = [
        ("Quantization Level", lambda s: str(s.get("quantization", "N/A"))),
        ("Model Size (Disk/RAM MB)", lambda s: f"{s.get('model_size_mb')} MB" if s.get('model_size_mb') is not None else "N/A"),
        ("VRAM Allocation", lambda s: str(s.get("vram_status", "Unavailable"))),
        ("Host System RAM Delta (MB)", lambda s: f"{s.get('system_ram_delta_mb'):+.1f} MB" if s.get('system_ram_delta_mb') is not None else "N/A"),
        ("Average Latency (s)", lambda s: f"{s.get('avg_latency_s', 0):.3f} s"),
        ("Aggregate Generation Tok/s", lambda s: f"{s.get('aggregate_generation_tokens_per_sec', 0):.2f} tok/s"),
        ("Total Tokens Generated", lambda s: str(s.get("total_output_tokens", 0))),
        ("Total Eval Duration (s)", lambda s: f"{s.get('total_eval_duration_s', 0):.3f} s"),
        ("Total Wall-Clock Time (s)", lambda s: f"{s.get('total_wall_clock_s', 0):.3f} s"),
    ]

    for label, extractor in metrics_spec:
        row = {"Metric": label}
        for m in models:
            s = results_map[m]["summary"]
            row[m] = extractor(s)
        csv_rows.append(row)

    comparison_json["comparison_table_rows"] = csv_rows
    return comparison_json, csv_rows


def print_comparison_table(results_map: Dict[str, Dict[str, Any]]) -> None:
    """Print an ASCII comparison table matching Requirement 8."""
    print("\n" + "=" * 90)
    print("OLLAMA BENCHMARK COMPARISON TABLE (Llama 3.2 1B Quantization Comparison)")
    print("=" * 90)

    header = f"{'Model':<30} {'Quant':<10} {'Size (MB)':>10} {'VRAM':>16} {'Avg Lat (s)':>12} {'Gen Tok/s':>12}"
    print(header)
    print("-" * 90)

    for model_name, data in results_map.items():
        s = data["summary"]
        quant = str(s.get("quantization", "N/A"))
        size_str = f"{s.get('model_size_mb', 'N/A')}"
        vram_mb = s.get("vram_mb")
        if vram_mb is not None and vram_mb > 0:
            vram_str = f"{vram_mb:.1f} MB"
        else:
            vram_str = "Unavailable"
        avg_lat = f"{s.get('avg_latency_s', 0):.3f}"
        agg_tps = f"{s.get('aggregate_generation_tokens_per_sec', 0):.2f}"

        print(f"{model_name:<30} {quant:<10} {size_str:>10} {vram_str:>16} {avg_lat:>12} {agg_tps:>12}")

    print("=" * 90)


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_file = os.path.join(base_dir, "prompts", "benchmark_prompts.json")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n========================================================")
    print("Task 3: Local Model Optimisation & Ollama Benchmarking")
    print("========================================================")

    # 1. Verify Ollama daemon availability
    if not check_ollama_running(OLLAMA_BASE_URL):
        print(f"\n[ERROR] Ollama daemon is NOT reachable at {OLLAMA_BASE_URL}.", file=sys.stderr)
        print("\nTroubleshooting Instructions (Windows PowerShell):", file=sys.stderr)
        print("  1. Verify Ollama is installed. If not, download from: https://ollama.ai", file=sys.stderr)
        print("  2. Start Ollama in background or separate shell:", file=sys.stderr)
        print("       ollama serve", file=sys.stderr)
        print("  3. Pull the required models:", file=sys.stderr)
        for m in TARGET_MODELS:
            print(f"       ollama pull {m}", file=sys.stderr)
        print("  4. Re-run this benchmark script.", file=sys.stderr)
        sys.exit(1)

    # 2. Check for required models in Ollama
    available_models = get_available_models(OLLAMA_BASE_URL)
    print(f"[Ollama] Reachable at {OLLAMA_BASE_URL}")
    print(f"[Ollama] Installed models in daemon: {available_models}")

    missing_models = [m for m in TARGET_MODELS if m not in available_models]
    if missing_models:
        print(f"\n[ERROR] The following target model(s) are missing from Ollama:", file=sys.stderr)
        for mm in missing_models:
            print(f"  - {mm}", file=sys.stderr)
        print("\nPlease pull the missing model(s) via PowerShell:", file=sys.stderr)
        for mm in missing_models:
            print(f"  ollama pull {mm}", file=sys.stderr)
        sys.exit(1)

    # 3. Load and strictly validate prompts
    if not os.path.exists(prompts_file):
        print(f"[ERROR] Prompts file not found at: {prompts_file}", file=sys.stderr)
        sys.exit(1)

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if not isinstance(prompts, list) or len(prompts) != 10:
        raise ValueError(
            f"Validation Failed: Expected exactly 10 benchmark prompts in {prompts_file}, "
            f"found {len(prompts) if isinstance(prompts, list) else type(prompts)}."
        )

    print(f"[Prompts] Loaded and validated exactly {len(prompts)} benchmark prompts.")

    # 4. Benchmark both target models
    results_map = {}
    for model_name in TARGET_MODELS:
        model_results = benchmark_single_model(model_name, prompts, OLLAMA_BASE_URL)
        results_map[model_name] = model_results

        # Save individual model outputs
        safe_name = model_name.replace(":", "_").replace("/", "_")
        single_out_path = os.path.join(results_dir, f"ollama_{safe_name}.json")
        with open(single_out_path, "w", encoding="utf-8") as f:
            json.dump(model_results, f, indent=2)
        print(f"  -> Saved individual output to: {single_out_path}")

    # 5. Generate and save combined JSON & CSV comparisons
    comparison_json, csv_rows = build_comparison_structures(results_map)

    json_cmp_path = os.path.join(results_dir, "ollama_comparison.json")
    with open(json_cmp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_json, f, indent=2)
    print(f"\n[Artifact] Combined comparison JSON saved: {json_cmp_path}")

    csv_cmp_path = os.path.join(results_dir, "ollama_comparison.csv")
    fieldnames = ["Metric"] + TARGET_MODELS
    with open(csv_cmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"[Artifact] Readable comparison CSV saved:  {csv_cmp_path}")

    # 6. Print final comparison table
    print_comparison_table(results_map)


if __name__ == "__main__":
    main()
