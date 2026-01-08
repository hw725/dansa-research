#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    args = ap.parse_args()

    dfa = _read(args.a)
    dfb = _read(args.b)

    print(f"A: {args.a} rows={len(dfa)} cols={len(dfa.columns)}")
    print(f"B: {args.b} rows={len(dfb)} cols={len(dfb.columns)}")
    print("A-only cols:", sorted(set(dfa.columns) - set(dfb.columns)))
    print("B-only cols:", sorted(set(dfb.columns) - set(dfa.columns)))

    print("\nA head:")
    print(dfa.head(3).to_string(index=False))
    print("\nB head:")
    print(dfb.head(3).to_string(index=False))

    # Basic sanity on key columns.
    for name, df in [("A", dfa), ("B", dfb)]:
        missing = [c for c in ["문단식별자", "원문", "번역문"] if c not in df.columns]
        if missing:
            print(f"[warn] {name} missing required cols: {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
