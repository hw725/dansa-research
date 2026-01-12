#!/usr/bin/env python3
"""Pick best config(s) from grid_search summary.json.

Primary objective: maximize micro_f1_tgt_exact.
Secondary objective: maximize mean_similarity.

Usage:
  py scripts/pick_best_from_summary.py test_results/grid_weights_100para_ex250_seed5/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: py scripts/pick_best_from_summary.py <summary.json>")
        return 2

    summary_path = Path(sys.argv[1])
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    results = data.get("results") or []
    if not results:
        raise SystemExit("summary.json has no results")

    rows = []
    for r in results:
        cfg = r.get("config") or {}
        seed_results = r.get("seed_results") or []
        # This runner stores one seed for this grid.
        if not seed_results:
            continue
        sr = seed_results[0]
        if not sr.get("success"):
            continue
        rows.append(
            {
                "config": cfg,
                "seed": int(sr["seed"]),
                "micro_f1_tgt_exact": float(sr["micro_f1_tgt_exact"]),
                "mean_similarity": float(sr.get("mean_similarity", 0.0)),
            }
        )

    if not rows:
        raise SystemExit("no successful runs")

    # Determine best by objectives
    best = max(rows, key=lambda x: (x["micro_f1_tgt_exact"], x["mean_similarity"]))

    best_f1 = best["micro_f1_tgt_exact"]
    best_rows = [r for r in rows if r["micro_f1_tgt_exact"] == best_f1]
    best_rows_sorted = sorted(best_rows, key=lambda x: x["mean_similarity"], reverse=True)

    out = {
        "best": best,
        "best_f1_ties": best_rows_sorted,
        "count": len(rows),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
