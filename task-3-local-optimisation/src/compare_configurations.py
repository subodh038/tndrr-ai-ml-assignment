"""
compare_configurations.py
=========================
Loads cpu_fp32_outputs.json and cpu_int8_outputs.json from the results/ directory,
prints a formatted comparison table, and saves results/comparison_table.json.
"""

import json
import os


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_speedup(base_val: float, cmp_val: float) -> float:
    """Return cmp_val / base_val (how many times faster/smaller cmp is vs base)."""
    if base_val == 0:
        return float("inf")
    return round(cmp_val / base_val, 3)


def print_table(fp32: dict, int8: dict) -> dict:
    s32 = fp32["summary"]
    s8 = int8["summary"]

    # Speedup: for time/latency lower is better → fp32/int8; for TPS higher is better → int8/fp32
    load_speedup   = round(s32["load_time_s"]       / s8["load_time_s"],       3) if s8["load_time_s"] > 0       else 0
    mem_speedup    = round(s32["model_memory_mb"]    / s8["model_memory_mb"],   3) if s8["model_memory_mb"] > 0   else 0
    tps_speedup    = round(s8["avg_tokens_per_sec"]  / s32["avg_tokens_per_sec"], 3) if s32["avg_tokens_per_sec"] > 0 else 0
    lat_speedup    = round(s32["avg_latency_s"]      / s8["avg_latency_s"],     3) if s8["avg_latency_s"] > 0     else 0

    rows = [
        ("Configuration", "Load Time",    "Memory (MB)",         "Avg TPS",                   "Avg Latency"),
        ("FP32",          f"{s32['load_time_s']:.3f}s", f"{s32['model_memory_mb']:.1f}", f"{s32['avg_tokens_per_sec']:.2f}", f"{s32['avg_latency_s']:.3f}s"),
        ("INT8 (dynamic)",f"{s8['load_time_s']:.3f}s",  f"{s8['model_memory_mb']:.1f}",  f"{s8['avg_tokens_per_sec']:.2f}",  f"{s8['avg_latency_s']:.3f}s"),
        ("Speedup (×)",   f"{load_speedup:.3f}×",        f"{mem_speedup:.3f}×",            f"{tps_speedup:.3f}×",             f"{lat_speedup:.3f}×"),
    ]

    col_widths = [max(len(r[i]) for r in rows) for i in range(5)]
    sep = "-+-".join("-" * w for w in col_widths)

    print()
    for i, row in enumerate(rows):
        print(" | ".join(cell.ljust(col_widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            print(sep)
    print()

    comparison = {
        "fp32": s32,
        "int8": s8,
        "speedup": {
            "load_time": load_speedup,
            "model_memory": mem_speedup,
            "avg_tokens_per_sec": tps_speedup,
            "avg_latency": lat_speedup,
        },
        "notes": (
            "Speedup for load_time / memory / latency: fp32/int8 (>1 means INT8 is faster/smaller). "
            "Speedup for TPS: int8/fp32 (>1 means INT8 is faster)."
        ),
    }
    return comparison


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")

    fp32_path = os.path.join(results_dir, "cpu_fp32_outputs.json")
    int8_path = os.path.join(results_dir, "cpu_int8_outputs.json")

    if not os.path.exists(fp32_path) or not os.path.exists(int8_path):
        print("ERROR: Run benchmark_cpu.py first to generate result files.")
        raise SystemExit(1)

    fp32 = load_result(fp32_path)
    int8 = load_result(int8_path)

    print("=== Configuration Comparison Table ===")
    comparison = print_table(fp32, int8)

    out_path = os.path.join(results_dir, "comparison_table.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {out_path}")
