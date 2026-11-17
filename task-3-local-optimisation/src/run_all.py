"""
run_all.py — Master runner for Task 3 (Local Optimisation)
===========================================================
Runs benchmark_cpu.py then compare_configurations.py in sequence,
and optionally runs analysis.py for the theoretical/measured comparison.
"""

import subprocess
import sys
import os


def run_script(script_path: str, cwd: str) -> int:
    print(f"\n{'='*60}")
    print(f"Running: {os.path.basename(script_path)}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=cwd,
    )
    return result.returncode


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir  = os.path.join(base_dir, "src")

    scripts = [
        os.path.join(src_dir, "benchmark_cpu.py"),
        os.path.join(src_dir, "compare_configurations.py"),
        os.path.join(src_dir, "analysis.py"),
    ]

    for script in scripts:
        rc = run_script(script, base_dir)
        if rc != 0:
            print(f"\nERROR: {script} exited with code {rc}")
            sys.exit(rc)

    print("\nAll Task 3 scripts completed successfully.")
