"""Spawn one subprocess per category for the query phase.

Each subprocess has its own Python interpreter, which means LightRAG's
JsonKVStorage singleton dict starts fresh. This eliminates the cross-instance
contamination that polluted the prior all-in-one-process run.

The inserted graph + VDB stay; only --only-query is run.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SCRIPT = HERE / "run_category.py"
CATEGORIES = ["II_니라_X", "IV_라_X", "I_니라_O", "III_라_O"]

if __name__ == "__main__":
    t_start = time.time()
    for cat in CATEGORIES:
        print(f"\n{'#'*70}\n# Subprocess for {cat}\n{'#'*70}", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            ["python.exe", "-u", str(SCRIPT), cat, "--only-query"],
            cwd=str(HERE),
        )
        print(f"# {cat} exited={proc.returncode} in {time.time()-t0:.1f}s", flush=True)
        if proc.returncode != 0:
            print(f"# WARN: {cat} subprocess failed; continuing", flush=True)
    print(f"\nALL DONE in {time.time()-t_start:.1f}s", flush=True)
