#!/usr/bin/env python3
"""PA 출력과 gold(문장 단위)를 비교해 번역문 불일치/문장수 차이를 디버깅한다.

- 기본 키: (book_name, 문단식별자)
- PA 출력에 book_name이 없으면 문단식별자만으로 비교(권장하지 않음)

예)
  docker-compose run --rm csp python scripts/debug_pa_vs_gold_mismatches.py \
    --pa-output test_results/pa_strict_20251227_023250.csv \
    --gold datasets/pa/test_100.csv \
    --out test_results/pa_vs_gold_mismatch_report.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from integrity_report import _norm  # 기존 정규화 규칙을 그대로 사용


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="PA vs gold mismatch reporter")
    ap.add_argument("--pa-output", required=True, type=str)
    ap.add_argument("--gold", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--limit", type=int, default=500, help="저장할 최대 행 수")
    args = ap.parse_args()

    pa_path = Path(args.pa_output)
    gold_path = Path(args.gold)
    out_path = Path(args.out)

    pred_df = _read(pa_path)
    gold_df = _read(gold_path)

    for col in ("문단식별자", "원문", "번역문"):
        if col not in pred_df.columns:
            raise SystemExit(f"PA 출력에 필수 컬럼이 없습니다: {col}")
    for col in ("문단식별자", "문장식별자", "원문", "번역문", "book_name"):
        if col not in gold_df.columns:
            raise SystemExit(f"gold에 필수 컬럼이 없습니다: {col}")

    pred_df = pred_df.copy()
    gold_df = gold_df.copy()

    pred_df["문단식별자"] = pred_df["문단식별자"].astype(int)
    gold_df["문단식별자"] = gold_df["문단식별자"].astype(int)

    pred_has_book = "book_name" in pred_df.columns
    if pred_has_book:
        pred_df["book_name"] = pred_df["book_name"].fillna("").astype(str)
    gold_df["book_name"] = gold_df["book_name"].fillna("").astype(str)

    if pred_has_book:
        pred_groups = pred_df.groupby(["book_name", "문단식별자"], sort=False)
        gold_groups = gold_df.sort_values(["book_name", "문단식별자", "문장식별자"], kind="stable").groupby(
            ["book_name", "문단식별자"], sort=False
        )
        keys = sorted(set(pred_groups.groups.keys()) & set(gold_groups.groups.keys()))
    else:
        pred_groups = pred_df.groupby(["문단식별자"], sort=False)
        gold_groups = gold_df.sort_values(["문단식별자", "문장식별자"], kind="stable").groupby(["문단식별자"], sort=False)
        keys = sorted(set(int(k) for k in pred_groups.groups.keys()) & set(int(k) for k in gold_groups.groups.keys()))

    rows: list[dict] = []

    for key in keys:
        if pred_has_book:
            bk, pid = key
            pred_g = pred_groups.get_group((bk, pid))
            gold_g = gold_groups.get_group((bk, pid))
        else:
            pid = int(key)
            bk = ""
            pred_g = pred_groups.get_group(pid)
            gold_g = gold_groups.get_group(pid)

        pred_tgt = [str(x).strip() for x in pred_g["번역문"].fillna("").tolist()]
        gold_tgt = [str(x).strip() for x in gold_g["번역문"].fillna("").tolist()]

        pred_tgt_norm = [_norm(s) for s in pred_tgt]
        gold_tgt_norm = [_norm(s) for s in gold_tgt]

        if pred_tgt_norm == gold_tgt_norm:
            continue

        mismatch_type = "content"
        if len(pred_tgt_norm) != len(gold_tgt_norm):
            mismatch_type = "count"

        # 첫 불일치 위치
        first_i = None
        for i, (a, b) in enumerate(zip(pred_tgt_norm, gold_tgt_norm)):
            if a != b:
                first_i = i
                break
        if first_i is None:
            first_i = min(len(pred_tgt_norm), len(gold_tgt_norm))

        def _safe_get(lst: list[str], i: int) -> str:
            if 0 <= i < len(lst):
                return lst[i]
            return ""

        rows.append(
            {
                "book_name": bk,
                "문단식별자": pid,
                "type": mismatch_type,
                "pred_n": len(pred_tgt_norm),
                "gold_n": len(gold_tgt_norm),
                "first_diff_i": first_i,
                "pred_norm": _safe_get(pred_tgt_norm, first_i)[:200],
                "gold_norm": _safe_get(gold_tgt_norm, first_i)[:200],
                "pred_raw": _safe_get(pred_tgt, first_i)[:200],
                "gold_raw": _safe_get(gold_tgt, first_i)[:200],
            }
        )

        if len(rows) >= int(args.limit):
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    print("=")
    print("PA vs gold mismatch report")
    print("=")
    print(f"PA output: {pa_path}")
    print(f"gold: {gold_path}")
    print(f"pred_has_book_name: {pred_has_book}")
    print(f"mismatch rows saved: {len(rows)}")
    print(f"out: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
