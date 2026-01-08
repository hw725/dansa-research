#!/usr/bin/env python3
"""번역문 문장리스트가 gold와 완전일치하는 문단들만 대상으로,
원문 경계(boundary)가 gold와 어디서/얼마나 다른지 디버깅한다.

핵심 의도
- 번역문이 이미 맞는 케이스들(subset)에서 원문 경계가 잘 끊겼는지 확인

출력
- boundary mismatch rows를 CSV로 저장 (기본 키: (book_name, 문단식별자))

예)
  docker-compose run --rm csp python scripts/debug_boundary_mismatches_tgt_exact.py \
    --pa-output test_results/sweep_runs/pa_strict_...seed2....csv \
    --gold datasets/pa/test_100.csv \
    --out test_results/boundary_mismatch_tgt_exact_seed2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from integrity_report import _boundary_positions_normed, _norm, _prf1


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _snippet_norm(full_norm: str, pos: int, radius: int = 40) -> str:
    if pos is None or pos < 0:
        return ""
    start = max(0, pos - radius)
    end = min(len(full_norm), pos + radius)
    return full_norm[start:end]


def main() -> int:
    ap = argparse.ArgumentParser(description="Boundary mismatch reporter on tgt-exact subset")
    ap.add_argument("--pa-output", required=True, type=str)
    ap.add_argument("--gold", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--limit", type=int, default=2000, help="저장할 최대 행 수")
    ap.add_argument("--radius", type=int, default=40, help="스니펫 반경(정규화 문자열 기준)")
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

    total = 0
    tgt_exact = 0
    rows: list[dict] = []

    # micro aggregates (tgt-exact subset only)
    tp = fp = fn = 0

    for key in keys:
        total += 1

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

        if pred_tgt_norm != gold_tgt_norm:
            continue

        tgt_exact += 1

        pred_src = [str(x).strip() for x in pred_g["원문"].fillna("").tolist()]
        gold_src = [str(x).strip() for x in gold_g["원문"].fillna("").tolist()]

        pred_b = _boundary_positions_normed(pred_src)
        gold_b = _boundary_positions_normed(gold_src)

        if pred_b == gold_b:
            continue

        inter = pred_b & gold_b
        tp_i = len(inter)
        fp_i = len(pred_b - gold_b)
        fn_i = len(gold_b - pred_b)
        tp += tp_i
        fp += fp_i
        fn += fn_i

        p_i, r_i, f1_i = _prf1(tp_i, fp_i, fn_i)

        sym = sorted(pred_b ^ gold_b)
        first_pos = sym[0] if sym else None

        pred_full_norm = "".join(_norm(s) for s in pred_src)
        gold_full_norm = "".join(_norm(s) for s in gold_src)

        rows.append(
            {
                "book_name": bk,
                "문단식별자": int(pid),
                "pred_n": len(pred_src),
                "gold_n": len(gold_src),
                "tp": tp_i,
                "fp": fp_i,
                "fn": fn_i,
                "precision": p_i,
                "recall": r_i,
                "f1": f1_i,
                "pred_boundaries": len(pred_b),
                "gold_boundaries": len(gold_b),
                "first_diff_pos_norm": first_pos,
                "first_pos_in_pred": (first_pos in pred_b) if first_pos is not None else None,
                "first_pos_in_gold": (first_pos in gold_b) if first_pos is not None else None,
                "pred_snip_norm": _snippet_norm(pred_full_norm, int(first_pos) if first_pos is not None else None, radius=int(args.radius)),
                "gold_snip_norm": _snippet_norm(gold_full_norm, int(first_pos) if first_pos is not None else None, radius=int(args.radius)),
            }
        )

        if len(rows) >= int(args.limit):
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    p_micro, r_micro, f1_micro = _prf1(tp, fp, fn)

    print("=")
    print("Boundary mismatch report (tgt exact subset)")
    print("=")
    print(f"PA output: {pa_path}")
    print(f"gold: {gold_path}")
    print(f"pred_has_book_name: {pred_has_book}")
    print(f"common keys: {len(keys)}")
    print(f"tgt_exact keys: {tgt_exact}")
    print(f"boundary mismatch rows saved: {len(df_out)}")
    print(f"micro P/R/F1 over tgt-exact subset: {p_micro:.4f} / {r_micro:.4f} / {f1_micro:.4f}")
    print(f"out: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
