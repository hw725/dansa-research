#!/usr/bin/env python3
"""Aggregate `summarize_pa_drift.py` output by threshold.

입력: pa_drift_summary_*.csv
출력: threshold별 집계(개수/중앙값/평균/최댓값 등) CSV

예)
  docker compose exec -T csp bash -lc "python -u scripts/aggregate_pa_drift_summary.py \
    --input test_results/pa_drift_summary_20251230_043230.csv"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _to_float(x: str | None) -> float | None:
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in {"none", "nan"}:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _to_int(x: str | None) -> int | None:
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in {"none", "nan"}:
        return None
    try:
        return int(float(x))
    except ValueError:
        return None


@dataclass
class Row:
    threshold: str
    micro_f1_tgt_exact: float | None
    micro_f1_all: float | None
    boundary_only_pids: int | None
    boundary_symdiff_mean: float | None
    translation_exact_ok: int | None


def _stats(values: list[float]) -> dict[str, float]:
    values_sorted = sorted(values)
    return {
        "min": values_sorted[0],
        "median": median(values_sorted),
        "mean": mean(values_sorted),
        "max": values_sorted[-1],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate pa_drift_summary by threshold")
    p.add_argument("--input", required=True, help="pa_drift_summary_*.csv (container 기준)")
    p.add_argument(
        "--out-csv",
        default=None,
        help="출력 CSV 경로(미지정 시 test_results/pa_drift_agg_<ts>.csv)",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = WORKSPACE_ROOT / in_path
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_csv) if args.out_csv else (WORKSPACE_ROOT / "test_results" / f"pa_drift_agg_{ts}.csv")

    buckets: dict[str, list[Row]] = defaultdict(list)

    with in_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit("empty csv")
        for d in r:
            thr = (d.get("threshold") or "").strip() or "(unknown)"
            buckets[thr].append(
                Row(
                    threshold=thr,
                    micro_f1_tgt_exact=_to_float(d.get("micro_f1_tgt_exact")),
                    micro_f1_all=_to_float(d.get("micro_f1_all")),
                    boundary_only_pids=_to_int(d.get("boundary_only_pids")),
                    boundary_symdiff_mean=_to_float(d.get("boundary_symdiff_mean")),
                    translation_exact_ok=_to_int(d.get("translation_exact_ok")),
                )
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "threshold",
        "n",
        "micro_f1_tgt_exact_min",
        "micro_f1_tgt_exact_median",
        "micro_f1_tgt_exact_mean",
        "micro_f1_tgt_exact_max",
        "micro_f1_all_median",
        "boundary_only_pids_median",
        "boundary_only_pids_max",
        "boundary_symdiff_mean_median",
        "translation_exact_ok_median",
    ]

    rows_out: list[dict[str, object]] = []

    for thr, rows in buckets.items():
        f1_ok = [x.micro_f1_tgt_exact for x in rows if x.micro_f1_tgt_exact is not None]
        f1_all = [x.micro_f1_all for x in rows if x.micro_f1_all is not None]
        boundary_only = [float(x.boundary_only_pids) for x in rows if x.boundary_only_pids is not None]
        symdiff_mean = [x.boundary_symdiff_mean for x in rows if x.boundary_symdiff_mean is not None]
        ok_cnt = [float(x.translation_exact_ok) for x in rows if x.translation_exact_ok is not None]

        if f1_ok:
            s = _stats(f1_ok)
            f1_min, f1_med, f1_mean, f1_max = s["min"], s["median"], s["mean"], s["max"]
        else:
            f1_min = f1_med = f1_mean = f1_max = None

        rows_out.append(
            {
                "threshold": thr,
                "n": len(rows),
                "micro_f1_tgt_exact_min": f1_min,
                "micro_f1_tgt_exact_median": f1_med,
                "micro_f1_tgt_exact_mean": f1_mean,
                "micro_f1_tgt_exact_max": f1_max,
                "micro_f1_all_median": (median(sorted(f1_all)) if f1_all else None),
                "boundary_only_pids_median": (median(sorted(boundary_only)) if boundary_only else None),
                "boundary_only_pids_max": (max(boundary_only) if boundary_only else None),
                "boundary_symdiff_mean_median": (median(sorted(symdiff_mean)) if symdiff_mean else None),
                "translation_exact_ok_median": (median(sorted(ok_cnt)) if ok_cnt else None),
            }
        )

    # micro_f1_tgt_exact_median desc -> max desc
    def _sort_key(d: dict[str, object]):
        med = d.get("micro_f1_tgt_exact_median")
        mx = d.get("micro_f1_tgt_exact_max")
        return (-(med if isinstance(med, (int, float)) and med is not None else -1), -(mx if isinstance(mx, (int, float)) and mx is not None else -1))

    rows_out.sort(key=_sort_key)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in rows_out:
            w.writerow(d)

    print(f"input: {in_path}")
    print(f"output: {out_path}")
    print("\nTop by micro_f1_tgt_exact_median:")
    for d in rows_out[:10]:
        print(
            f"  thr={d['threshold']} n={d['n']} "
            f"med={d['micro_f1_tgt_exact_median']} max={d['micro_f1_tgt_exact_max']} "
            f"boundary_only_med={d['boundary_only_pids_median']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
