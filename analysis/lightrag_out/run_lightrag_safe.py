"""Subprocess-isolated LightRAG runner. One process per category → no memory leak.

Usage:
    py -3.13 run_lightrag_safe.py              # all 4 categories
    py -3.13 run_lightrag_safe.py I_니라_O      # specific category
    py -3.13 run_lightrag_safe.py --only-query  # skip insert, query only
"""
import subprocess
import sys
import time
import json
from pathlib import Path

ROOT = Path(r"C:/Users/junto/Downloads/analysis_v8")
OUT = ROOT / "lightrag_out"
RESULTS = OUT / "results"
PYTHON = [sys.executable]  # reuse whatever interpreter launched this script

CATEGORIES = ["I_니라_O", "II_니라_X", "III_라_O", "IV_라_X"]
MAX_RETRIES = 1


def run_category(cat: str, only_query: bool = False) -> tuple[bool, float]:
    log_file = OUT / f"log_{cat}.txt"
    args = PYTHON + [str(OUT / "run_lightrag.py"), cat]
    if only_query:
        args.append("--only-query")

    print(f"\n{'='*60}", flush=True)
    print(f"  [{cat}] starting (subprocess isolated)", flush=True)
    print(f"  log: {log_file}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            if any(kw in line for kw in ("insert done", "query ", "[skip]", "ERROR", "FAIL", "Traceback")):
                print(f"  {line.rstrip()}", flush=True)

        proc.wait()

    elapsed = time.time() - t0
    success = proc.returncode == 0

    if success:
        print(f"  [{cat}] completed in {elapsed:.0f}s", flush=True)
    else:
        print(f"  [{cat}] FAILED (exit={proc.returncode}) after {elapsed:.0f}s", flush=True)
        print(f"  see log: {log_file}", flush=True)

    return success, elapsed


def main():
    only_query = "--only-query" in sys.argv
    cli_cats = [a for a in sys.argv[1:] if not a.startswith("--")]
    cats = cli_cats if cli_cats else CATEGORIES

    samples_path = OUT / "cluster_full_v2.json"
    if not samples_path.exists():
        print(f"ERROR: {samples_path} not found", flush=True)
        sys.exit(1)

    print(f"Categories: {cats}  (only_query={only_query})", flush=True)
    print(f"Each category runs in its own subprocess to prevent memory leaks.", flush=True)

    report = {}
    t_total = time.time()

    for cat in cats:
        success, elapsed = run_category(cat, only_query)

        if not success and MAX_RETRIES > 0:
            print(f"  [{cat}] retrying (1/{MAX_RETRIES})...", flush=True)
            success, elapsed2 = run_category(cat, only_query)
            elapsed += elapsed2

        report[cat] = {"success": success, "elapsed_s": round(elapsed, 1)}

        if not success:
            print(f"  [{cat}] giving up after retry. Continuing to next category.", flush=True)

    total_elapsed = time.time() - t_total

    print(f"\n{'='*60}", flush=True)
    print(f"  SUMMARY  (total: {total_elapsed:.0f}s)", flush=True)
    print(f"{'='*60}", flush=True)
    for cat, r in report.items():
        status = "OK" if r["success"] else "FAIL"
        print(f"  {cat}: {status} ({r['elapsed_s']}s)", flush=True)

    # Check all query results exist
    expected_queries = ["Q1_themes", "Q2_cluster_diff", "Q3_decision_features",
                        "Q4_logical_markers", "Q5_tense_mood", "Q6_sources"]
    missing = []
    for cat in cats:
        for qid in expected_queries:
            f = RESULTS / f"{cat}__{qid}.md"
            if not f.exists() or f.stat().st_size < 200:
                missing.append(f"{cat}/{qid}")

    if missing:
        print(f"\n  WARNING: {len(missing)} missing/incomplete results:", flush=True)
        for m in missing:
            print(f"    - {m}", flush=True)
    else:
        print(f"\n  All {len(cats) * len(expected_queries)} query results present.", flush=True)

    all_ok = all(r["success"] for r in report.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
