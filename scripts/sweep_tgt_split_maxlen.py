#!/usr/bin/env python3
"""Sweep split_target_sentences_advanced max_length for gold tgt exact.

This is a lightweight analysis tool (no training).

Usage:
  python scripts/sweep_tgt_split_maxlen.py --pd datasets/pd/test_100.csv --gold datasets/pa/test_100_from_pd.csv --lengths 120 150 180 220 260 320 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pd", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--lengths", nargs="+", type=int, required=True)
    args = p.parse_args()

    workspace_root = Path(__file__).resolve().parents[1]
    pa_dir = workspace_root / "pa"
    sys.path.insert(0, str(workspace_root))
    sys.path.insert(0, str(pa_dir))

    from sentence_splitter import split_target_sentences_advanced

    df_pd = _load(Path(args.pd)).copy()
    df_gold = _load(Path(args.gold)).copy()

    for df in (df_pd, df_gold):
        df["book_name"] = df["book_name"].astype(str)
        df["문단식별자"] = df["문단식별자"].astype(int)
        df["번역문"] = df["번역문"].astype(str)

    pd_map = {
        (str(r["book_name"]), int(r["문단식별자"])): str(r["번역문"])
        for _, r in df_pd.iterrows()
    }
    gold_grp = df_gold.groupby(["book_name", "문단식별자"], sort=False)["번역문"].apply(list)

    def norm(x: str) -> str:
        return str(x).replace("\r\n", "\n").strip()

    total = len(gold_grp)
    for L in args.lengths:
        exact = 0
        for key, gold_list in gold_grp.items():
            text = pd_map.get(key, "")
            pred = [norm(s) for s in split_target_sentences_advanced(text, max_length=int(L))]
            gold = [norm(x) for x in gold_list]
            if pred == gold:
                exact += 1
        print(f"max_length={L}\ttranslation_exact={exact}/{total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
