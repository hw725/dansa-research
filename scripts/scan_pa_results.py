#!/usr/bin/env python3
"""Scan PA prediction CSVs against a fixed gold and report scores.

Purpose: reliably match "which pred file produced which score" without
PowerShell heredoc/quoting issues.

Defaults are tuned for this repo:
- pred glob: test_results/pa_strict*.csv
- gold:      datasets/pa/test_100.csv

This script intentionally re-implements the metric logic from
integrity_report.run_pa_output_vs_gold_report(), but returns machine-readable
rows instead of printing long per-file reports.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

# Ensure repo root is importable when running from scripts/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from integrity_report import _boundary_positions_normed, _norm, _prf1


def _read_csv_robust(path: Path) -> pd.DataFrame:
    # Files in this repo are typically UTF-8 with BOM.
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


@dataclass(frozen=True)
class ScoreRow:
    pred_file: str
    pred_has_book: bool
    total_paras: int
    tgt_exact_ok: int
    micro_f1_all: float
    micro_f1_tgt_exact: float


def score_one(pred_path: Path, gold_path: Path) -> ScoreRow:
    pred_df = _read_csv_robust(pred_path)
    gold_df = _read_csv_robust(gold_path)

    required_pred = {"문단식별자", "원문", "번역문"}
    required_gold = {"문단식별자", "문장식별자", "원문", "번역문", "book_name"}

    missing_pred = sorted(required_pred - set(pred_df.columns))
    if missing_pred:
        raise ValueError(f"pred missing columns: {missing_pred}")

    missing_gold = sorted(required_gold - set(gold_df.columns))
    if missing_gold:
        raise ValueError(f"gold missing columns: {missing_gold}")

    pred_df = pred_df.copy()
    gold_df = gold_df.copy()

    for col in ("원문", "번역문"):
        pred_df[col] = pred_df[col].fillna("")
        gold_df[col] = gold_df[col].fillna("")
    gold_df["book_name"] = gold_df["book_name"].fillna("")

    pred_df["문단식별자"] = pred_df["문단식별자"].astype(int)
    gold_df["문단식별자"] = gold_df["문단식별자"].astype(int)

    pred_has_book = "book_name" in pred_df.columns
    if pred_has_book:
        pred_df["book_name"] = pred_df["book_name"].fillna("").astype(str)

    if pred_has_book:
        pred_groups = pred_df.groupby(["book_name", "문단식별자"], sort=False)
        gold_groups = (
            gold_df.sort_values(["book_name", "문단식별자", "문장식별자"], kind="stable")
            .groupby(["book_name", "문단식별자"], sort=False)
        )
        pred_keys = set(pred_groups.groups.keys())
        gold_keys = set(gold_groups.groups.keys())
        common_keys = sorted(pred_keys & gold_keys)
    else:
        pred_groups = pred_df.groupby("문단식별자", sort=False)
        gold_groups = (
            gold_df.sort_values(["문단식별자", "문장식별자"], kind="stable")
            .groupby("문단식별자", sort=False)
        )
        pred_keys = set(int(k) for k in pred_groups.groups.keys())
        gold_keys = set(int(k) for k in gold_groups.groups.keys())
        common_keys = sorted(pred_keys & gold_keys)

    tp = fp = fn = 0
    tp_ok = fp_ok = fn_ok = 0
    tgt_exact_ok = 0

    for key in common_keys:
        if pred_has_book:
            bk, pid = key
            pred_g = pred_groups.get_group((bk, pid))
            gold_g = gold_groups.get_group((bk, pid))
        else:
            pid = int(key)
            pred_g = pred_groups.get_group(pid)
            gold_g = gold_groups.get_group(pid)

        pred_src = [str(x).strip() for x in pred_g["원문"].tolist()]
        pred_tgt = [str(x).strip() for x in pred_g["번역문"].tolist()]
        gold_src = [str(x).strip() for x in gold_g["원문"].tolist()]
        gold_tgt = [str(x).strip() for x in gold_g["번역문"].tolist()]

        pred_tgt_norm = [_norm(s) for s in pred_tgt]
        gold_tgt_norm = [_norm(s) for s in gold_tgt]
        tgt_match = pred_tgt_norm == gold_tgt_norm
        if tgt_match:
            tgt_exact_ok += 1

        pred_b = _boundary_positions_normed(pred_src)
        gold_b = _boundary_positions_normed(gold_src)
        inter = pred_b & gold_b

        tp_i = len(inter)
        fp_i = len(pred_b - gold_b)
        fn_i = len(gold_b - pred_b)

        tp += tp_i
        fp += fp_i
        fn += fn_i

        if tgt_match:
            tp_ok += tp_i
            fp_ok += fp_i
            fn_ok += fn_i

    _, _, f1_all = _prf1(tp, fp, fn)
    _, _, f1_ok = _prf1(tp_ok, fp_ok, fn_ok)

    return ScoreRow(
        pred_file=pred_path.name,
        pred_has_book=pred_has_book,
        total_paras=len(common_keys),
        tgt_exact_ok=tgt_exact_ok,
        micro_f1_all=float(f1_all),
        micro_f1_tgt_exact=float(f1_ok),
    )


def _iter_pred_files(root: Path, pattern: str) -> Iterable[Path]:
    # Allow either a glob pattern or a direct file path.
    p = (root / pattern)
    if p.exists() and p.is_file():
        yield p
        return

    # If pattern contains path separators, interpret as a path glob.
    # Otherwise, search under root.
    if any(sep in pattern for sep in ["/", "\\"]):
        yield from root.glob(pattern)
    else:
        yield from (root / "test_results").glob(pattern)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root (default: auto-detected)",
    )
    ap.add_argument(
        "--gold",
        type=str,
        default="datasets/pa/test_100.csv",
        help="Gold CSV relative to root",
    )
    ap.add_argument(
        "--pred-glob",
        type=str,
        default="pa_strict*.csv",
        help="Glob (default: pa_strict*.csv under test_results)",
    )
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument(
        "--targets",
        type=float,
        nargs="*",
        default=[0.4337, 0.4177],
        help="Target micro_f1_tgt_exact values to find nearest files",
    )
    args = ap.parse_args()

    root = args.root
    gold_path = root / args.gold
    if not gold_path.exists():
        raise SystemExit(f"gold not found: {gold_path}")

    pred_files = sorted(_iter_pred_files(root, args.pred_glob))
    if not pred_files:
        raise SystemExit(f"no pred files matched: {args.pred_glob}")

    rows: list[ScoreRow] = []
    failures: list[tuple[str, str]] = []

    for pred_path in pred_files:
        try:
            rows.append(score_one(pred_path, gold_path))
        except Exception as e:  # noqa: BLE001
            failures.append((pred_path.name, str(e)))

    rows.sort(key=lambda r: (r.micro_f1_tgt_exact, r.micro_f1_all), reverse=True)

    print(f"gold = {gold_path}")
    print(f"pred_files = {len(pred_files)}, scored = {len(rows)}, failed = {len(failures)}")
    print()

    print(f"TOP {args.top} by micro_f1_tgt_exact")
    for r in rows[: args.top]:
        has_book = "book" if r.pred_has_book else "pid-only"
        print(
            f"{r.pred_file}\t{has_book}\tparas={r.total_paras}\t"
            f"tgt_exact={r.tgt_exact_ok}/{r.total_paras}\t"
            f"micro_f1_all={r.micro_f1_all:.4f}\t"
            f"micro_f1_tgt_exact={r.micro_f1_tgt_exact:.4f}"
        )

    if args.targets and rows:
        print()
        for t in args.targets:
            best = min(rows, key=lambda r: abs(r.micro_f1_tgt_exact - float(t)))
            has_book = "book" if best.pred_has_book else "pid-only"
            print(
                f"closest_to_{t:.4f}: {best.pred_file} ({has_book}, "
                f"micro_f1_tgt_exact={best.micro_f1_tgt_exact:.4f}, micro_f1_all={best.micro_f1_all:.4f}, "
                f"tgt_exact={best.tgt_exact_ok}/{best.total_paras})"
            )

    if failures:
        print() 
        print("FAILURES (first 20):")
        for name, msg in failures[:20]:
            print(f"- {name}: {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
