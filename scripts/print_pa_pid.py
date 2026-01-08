#!/usr/bin/env python3
"""Print PA output rows for a specific (book_name, 문단식별자).

This is a tiny inspection helper to compare how a paragraph was split across
runs without any hardcoded exceptions.

Example (docker):
  docker-compose run --rm csp python scripts/print_pa_pid.py \
    test_results/pa_strict_inputPD_thr0.7_ml10_seed1_refined_v5new.csv \
    --book "당송팔대가문초구양수1" --pid 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--book", default=None, help="book_name (optional if csv lacks book_name)")
    ap.add_argument("--pid", type=int, required=True, help="문단식별자")
    ap.add_argument(
        "--cols",
        nargs="*",
        default=["book_name", "문단식별자", "문장식별자", "원문", "번역문", "similarity"],
    )
    args = ap.parse_args()

    df = _read_csv(args.csv)

    if "문단식별자" not in df.columns:
        raise SystemExit("csv missing column: 문단식별자")
    df = df.copy()
    df["문단식별자"] = df["문단식별자"].astype(int)

    mask = df["문단식별자"] == int(args.pid)

    has_book = "book_name" in df.columns
    if has_book:
        if args.book is None:
            raise SystemExit("csv has book_name; please pass --book")
        mask = mask & (df["book_name"].fillna("").astype(str) == str(args.book))

    sub = df.loc[mask].copy()
    if sub.empty:
        raise SystemExit("no rows matched")

    sort_cols = [c for c in ["book_name", "문단식별자", "문장식별자"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols, kind="stable")

    cols = [c for c in args.cols if c in sub.columns]
    print(f"file={args.csv} rows={len(sub)}")
    if has_book:
        print(f"key=({args.book}, {args.pid})")
    else:
        print(f"key=({args.pid})")
    print(sub[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
