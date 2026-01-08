#!/usr/bin/env python3
"""Debug boundary differences between PA output and gold for a single (book_name, pid).

- Uses the same normalization/boundary logic as integrity_report.py
- Prints sentence lists (src), boundary sets, and a short context snippet around differing boundaries.

Example (PowerShell):
  docker-compose run --rm csp python scripts/debug_pa_boundary_diff.py \
    --pred test_results/pa_best_thr0.7_ml10_seed1.csv \
    --gold datasets/pa/test_100.csv \
    --book-name "당송팔대가문초구양수1" \
    --pid 353
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Import the exact functions used in evaluation.
# Ensure repo root is importable when running as `python scripts/...py`
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from integrity_report import _norm, _boundary_positions_normed  # type: ignore


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_excel(p)


def _group(df: pd.DataFrame, *, book: str | None, pid: int):
    if "book_name" in df.columns and book is not None:
        return df[(df["book_name"].astype(str) == str(book)) & (df["문단식별자"].astype(int) == int(pid))]
    return df[df["문단식별자"].astype(int) == int(pid)]


def _ctx(norm_joined: str, pos: int, radius: int = 25) -> str:
    a = max(0, pos - radius)
    b = min(len(norm_joined), pos + radius)
    return norm_joined[a:b]


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug PA boundary diffs for one paragraph")
    ap.add_argument("--pred", required=True, help="PA output CSV/XLSX")
    ap.add_argument("--gold", required=True, help="gold CSV/XLSX")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--book-name", default=None, help="book_name (recommended if available)")
    ap.add_argument("--show", type=int, default=10, help="how many boundary diffs to show")
    args = ap.parse_args()

    pred = _read(args.pred)
    gold = _read(args.gold)

    pred_g = _group(pred, book=args.book_name, pid=args.pid)
    gold_g = _group(gold, book=args.book_name, pid=args.pid)

    if pred_g.empty:
        raise SystemExit(f"pred에서 해당 key를 찾지 못함: book={args.book_name} pid={args.pid}")
    if gold_g.empty:
        raise SystemExit(f"gold에서 해당 key를 찾지 못함: book={args.book_name} pid={args.pid}")

    pred_src = [str(x) for x in pred_g["원문"].tolist()]
    gold_src = [str(x) for x in gold_g["원문"].tolist()]

    pred_b = sorted(_boundary_positions_normed(pred_src))
    gold_b = sorted(_boundary_positions_normed(gold_src))

    pred_join = "".join(_norm(s) for s in pred_src)
    gold_join = "".join(_norm(s) for s in gold_src)

    print("=" * 120)
    print(f"KEY: book_name={args.book_name} pid={args.pid}")
    print(f"pred sentences: {len(pred_src)} | gold sentences: {len(gold_src)}")
    print(f"pred boundaries: {len(pred_b)} | gold boundaries: {len(gold_b)}")

    inter = set(pred_b) & set(gold_b)
    only_pred = sorted(set(pred_b) - set(gold_b))
    only_gold = sorted(set(gold_b) - set(pred_b))

    print(f"intersection: {len(inter)}")
    print(f"only_pred: {len(only_pred)} | only_gold: {len(only_gold)}")

    print("\n-- pred src sentences (norm-len) --")
    for i, s in enumerate(pred_src, start=1):
        print(f"P{i:02d} len={len(_norm(s))}: {s}")

    print("\n-- gold src sentences (norm-len) --")
    for i, s in enumerate(gold_src, start=1):
        print(f"G{i:02d} len={len(_norm(s))}: {s}")

    def _show_list(title: str, positions: list[int], joined: str):
        print(f"\n-- {title} (show up to {args.show}) --")
        for pos in positions[: args.show]:
            print(f"pos={pos} ctx={_ctx(joined, pos)}")

    _show_list("only_pred boundaries", only_pred, pred_join)
    _show_list("only_gold boundaries", only_gold, gold_join)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
