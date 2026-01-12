#!/usr/bin/env python3
"""Select a representative seed from grid_search_pa_weights.py output.

Rule (as agreed): choose the median seed by micro_f1_tgt_exact across the given seeds.
Ties are broken by closeness to the mean, then by seed id.

Usage:
  py scripts/select_representative_seed.py test_results/seed_calibration_100para_ex250_s1_5/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: py scripts/select_representative_seed.py <summary.json>")
        return 2

    summary_path = Path(sys.argv[1])
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    results = data.get("results") or []
    if not results:
        raise SystemExit("summary.json has no results")

    seed_results = (results[0].get("seed_results") or [])
    seed_results = [sr for sr in seed_results if sr.get("success")]
    if not seed_results:
        raise SystemExit("no successful seeds")

    f1s = [float(sr["micro_f1_tgt_exact"]) for sr in seed_results]
    mean_f1 = sum(f1s) / len(f1s)

    # median by F1
    seed_results_sorted = sorted(seed_results, key=lambda sr: (float(sr["micro_f1_tgt_exact"]), int(sr["seed"])))
    median_idx = len(seed_results_sorted) // 2
    median_f1 = float(seed_results_sorted[median_idx]["micro_f1_tgt_exact"])

    median_candidates = [
        sr for sr in seed_results_sorted if float(sr["micro_f1_tgt_exact"]) == median_f1
    ]

    chosen = sorted(
        median_candidates,
        key=lambda sr: (abs(float(sr["micro_f1_tgt_exact"]) - mean_f1), int(sr["seed"]))
    )[0]

    print(json.dumps({
        "chosen_seed": int(chosen["seed"]),
        "chosen_micro_f1_tgt_exact": float(chosen["micro_f1_tgt_exact"]),
        "chosen_mean_similarity": float(chosen.get("mean_similarity", 0.0)),
        "mean_micro_f1_tgt_exact": mean_f1,
        "median_micro_f1_tgt_exact": median_f1,
        "all": [
            {
                "seed": int(sr["seed"]),
                "micro_f1_tgt_exact": float(sr["micro_f1_tgt_exact"]),
                "mean_similarity": float(sr.get("mean_similarity", 0.0)),
            }
            for sr in seed_results_sorted
        ],
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
