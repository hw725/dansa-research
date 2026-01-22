#!/usr/bin/env python3
"""Diff two P2S output CSVs by (book_name, 문단식별자).

Goal: pinpoint which pids differ between two outputs (e.g., v5new vs rerun)
without any hardcoded exceptions.

Outputs:
- missing keys (A-only / B-only)
- per-pid differences: tgt sentence-list mismatch, src boundary-set mismatch
- "first" mismatches in pid order

Optionally accepts --gold to also compute per-pid boundary F1 vs gold (same as
p2s_evaluator.run_p2s_output_vs_gold_report logic).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# Allow running this script from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from accuracy.p2s_evaluator import _boundary_positions_normed, _norm, _prf1


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def _prep(df: pd.DataFrame, *, name: str) -> tuple[pd.DataFrame, bool]:
    required = {"문단식별자", "원문", "번역문"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{name} missing columns: {missing}")

    df = df.copy()
    df["문단식별자"] = df["문단식별자"].astype(int)
    for col in ("원문", "번역문"):
        df[col] = df[col].fillna("")

    has_book = "book_name" in df.columns
    if has_book:
        df["book_name"] = df["book_name"].fillna("").astype(str)

    return df, has_book


def _group(df: pd.DataFrame, has_book: bool):
    if has_book:
        return df.groupby(["book_name", "문단식별자"], sort=False)
    return df.groupby("문단식별자", sort=False)


def _key_sort(key):
    # key is (book, pid) or pid
    if isinstance(key, tuple):
        bk, pid = key
        return (str(bk), int(pid))
    return ("", int(key))


@dataclass(frozen=True)
class DiffRow:
    key: object
    a_n: int
    b_n: int
    tgt_equal: bool
    src_equal: bool
    src_concat_equal: bool
    boundary_equal: bool
    boundary_symdiff: int
    f1_gold_a: float | None
    f1_gold_b: float | None


def _seq_norm(xs: Iterable[str]) -> list[str]:
    return [_norm(str(x).strip()) for x in xs]


def _f1_vs_gold(pred_src: list[str], gold_src: list[str]) -> float:
    pb = _boundary_positions_normed(pred_src)
    gb = _boundary_positions_normed(gold_src)
    inter = pb & gb
    tp = len(inter)
    fp = len(pb - gb)
    fn = len(gb - pb)
    _, _, f1 = _prf1(tp, fp, fn)
    return float(f1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path, help="P2S output A (csv)")
    ap.add_argument("b", type=Path, help="P2S output B (csv)")
    ap.add_argument("--gold", type=Path, default=None, help="Optional gold sentence CSV")
    ap.add_argument("--top", type=int, default=30, help="How many differing pids to print")
    args = ap.parse_args()

    a_df, a_has_book = _prep(_read_csv(args.a), name="A")
    b_df, b_has_book = _prep(_read_csv(args.b), name="B")

    # If one has book_name and the other doesn't, we still can diff by pid-only,
    # but warn since pid collisions are possible.
    if a_has_book != b_has_book:
        print("[warn] A/B book_name column presence differs. Falling back to pid-only may be ambiguous.")

    use_book = a_has_book and b_has_book

    a_g = _group(a_df, use_book)
    b_g = _group(b_df, use_book)

    a_keys = set(a_g.groups.keys())
    b_keys = set(b_g.groups.keys())

    only_a = sorted(a_keys - b_keys, key=_key_sort)
    only_b = sorted(b_keys - a_keys, key=_key_sort)
    common = sorted(a_keys & b_keys, key=_key_sort)

    gold_g = None
    if args.gold is not None:
        gold_df = _read_csv(args.gold)
        required_gold = {"문단식별자", "문장식별자", "원문", "번역문", "book_name"}
        missing_gold = sorted(required_gold - set(gold_df.columns))
        if missing_gold:
            raise SystemExit(f"gold missing columns: {missing_gold}")
        gold_df = gold_df.copy()
        gold_df["문단식별자"] = gold_df["문단식별자"].astype(int)
        for col in ("원문", "번역문"):
            gold_df[col] = gold_df[col].fillna("")
        gold_df["book_name"] = gold_df["book_name"].fillna("").astype(str)
        gold_g = gold_df.sort_values(["book_name", "문단식별자", "문장식별자"], kind="stable").groupby(
            ["book_name", "문단식별자"], sort=False
        )

    diffs: list[DiffRow] = []

    for key in common:
        a_grp = a_g.get_group(key)
        b_grp = b_g.get_group(key)

        a_src = [str(x).strip() for x in a_grp["원문"].tolist()]
        b_src = [str(x).strip() for x in b_grp["원문"].tolist()]
        a_tgt = [str(x).strip() for x in a_grp["번역문"].tolist()]
        b_tgt = [str(x).strip() for x in b_grp["번역문"].tolist()]

        tgt_equal = _seq_norm(a_tgt) == _seq_norm(b_tgt)
        src_equal = _seq_norm(a_src) == _seq_norm(b_src)
        src_concat_equal = _norm("".join(a_src)) == _norm("".join(b_src))

        a_b = _boundary_positions_normed(a_src)
        b_b = _boundary_positions_normed(b_src)
        boundary_equal = a_b == b_b
        boundary_symdiff = len(a_b ^ b_b)

        f1a = f1b = None
        if gold_g is not None:
            try:
                gold_grp = gold_g.get_group(key)
                gold_src = [str(x).strip() for x in gold_grp["원문"].tolist()]
                f1a = _f1_vs_gold(a_src, gold_src)
                f1b = _f1_vs_gold(b_src, gold_src)
            except KeyError:
                pass

        if (not tgt_equal) or (not boundary_equal) or (len(a_src) != len(b_src)):
            diffs.append(
                DiffRow(
                    key=key,
                    a_n=len(a_src),
                    b_n=len(b_src),
                    tgt_equal=tgt_equal,
                    src_equal=src_equal,
                    src_concat_equal=src_concat_equal,
                    boundary_equal=boundary_equal,
                    boundary_symdiff=boundary_symdiff,
                    f1_gold_a=f1a,
                    f1_gold_b=f1b,
                )
            )

    print("=")
    print(f"A: {args.a}")
    print(f"B: {args.b}")
    if args.gold is not None:
        print(f"gold: {args.gold}")
    print(f"keys: common={len(common)}, onlyA={len(only_a)}, onlyB={len(only_b)}")

    if only_a:
        print("only in A (first 10):", ", ".join(str(k) for k in only_a[:10]))
    if only_b:
        print("only in B (first 10):", ", ".join(str(k) for k in only_b[:10]))

    print(f"diff pids: {len(diffs)}")
    if diffs:
        tgt_same = sum(1 for r in diffs if r.tgt_equal)
        src_concat_same = sum(1 for r in diffs if r.src_concat_equal)
        print(f"diff breakdown: tgt_equal={tgt_same}/{len(diffs)}, src_concat_equal={src_concat_same}/{len(diffs)}")

    # Print the earliest diffs in pid order.
    diffs_sorted = sorted(diffs, key=lambda r: _key_sort(r.key))
    print()
    print(f"FIRST {min(args.top, len(diffs_sorted))} diffs (pid order)")
    for r in diffs_sorted[: args.top]:
        key_s = str(r.key)
        extra = ""
        if (r.f1_gold_a is not None) and (r.f1_gold_b is not None):
            extra = f"\tf1_gold: A={r.f1_gold_a:.4f} B={r.f1_gold_b:.4f} Δ={(r.f1_gold_b - r.f1_gold_a):+.4f}"
        print(
            f"{key_s}\ta_n={r.a_n}\tb_n={r.b_n}\t"
            f"tgt_equal={int(r.tgt_equal)}\tsrc_equal={int(r.src_equal)}\t"
            f"src_concat_equal={int(r.src_concat_equal)}\t"
            f"boundary_equal={int(r.boundary_equal)}\tboundary_symdiff={r.boundary_symdiff}{extra}"
        )

    # Print a few concrete examples for the very first diff.
    if diffs_sorted:
        first = diffs_sorted[0].key
        print() 
        print("EXAMPLE (first diff):", first)
        a_grp = a_g.get_group(first)
        b_grp = b_g.get_group(first)
        a_tgt = [str(x).strip() for x in a_grp["번역문"].tolist()]
        b_tgt = [str(x).strip() for x in b_grp["번역문"].tolist()]
        a_src = [str(x).strip() for x in a_grp["원문"].tolist()]
        b_src = [str(x).strip() for x in b_grp["원문"].tolist()]

        if _seq_norm(a_tgt) != _seq_norm(b_tgt):
            print("- tgt differs")
            for i, (ta, tb) in enumerate(zip(a_tgt, b_tgt)):
                if _norm(ta) != _norm(tb):
                    print(f"  first tgt mismatch at idx={i}")
                    print(f"  A: {ta}")
                    print(f"  B: {tb}")
                    break
            else:
                if len(a_tgt) != len(b_tgt):
                    print(f"  tgt length differs: A={len(a_tgt)} B={len(b_tgt)}")

        if _boundary_positions_normed(a_src) != _boundary_positions_normed(b_src):
            print("- src boundary differs")
            ab = sorted(_boundary_positions_normed(a_src))
            bb = sorted(_boundary_positions_normed(b_src))
            print(f"  A boundaries (first 20): {ab[:20]}")
            print(f"  B boundaries (first 20): {bb[:20]}")

        if _seq_norm(a_src) != _seq_norm(b_src):
            print("- src text differs")
            for i, (sa, sb) in enumerate(zip(a_src, b_src)):
                if _norm(sa) != _norm(sb):
                    print(f"  first src mismatch at idx={i}")
                    print(f"  A: {sa}")
                    print(f"  B: {sb}")
                    break
            else:
                if len(a_src) != len(b_src):
                    print(f"  src length differs: A={len(a_src)} B={len(b_src)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
