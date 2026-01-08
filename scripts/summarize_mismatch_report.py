#!/usr/bin/env python3
"""Mismatch report CSV를 요약 출력한다.

입력은 scripts/debug_pa_vs_gold_mismatches.py가 생성한 CSV 형식을 가정한다.

예)
  docker-compose run --rm csp python scripts/summarize_mismatch_report.py \
    --report test_results/pa_vs_gold_mismatch_report.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize PA vs gold mismatch report")
    ap.add_argument("--report", required=True, type=str)
    ap.add_argument("--top", type=int, default=10, help="예시로 보여줄 최대 행 수")
    args = ap.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(f"report 파일이 존재하지 않습니다: {report_path}")

    df = pd.read_csv(report_path)
    if len(df) == 0:
        print("mismatch rows: 0 (완전일치)")
        return 0

    required = {"type", "pred_n", "gold_n", "first_diff_i", "문단식별자"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"리포트에 필수 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["type"] = df["type"].fillna("").astype(str)
    df["first_diff_i"] = pd.to_numeric(df["first_diff_i"], errors="coerce")

    print("=")
    print("Mismatch summary")
    print("=")
    print(f"report: {report_path}")
    print(f"mismatch rows: {len(df)}")

    type_counts = df["type"].value_counts(dropna=False)
    print("\n[type]")
    for k, v in type_counts.items():
        print(f"- {k}: {int(v)}")

    # 문장 수 차이 통계
    if "pred_n" in df.columns and "gold_n" in df.columns:
        df["n_delta"] = pd.to_numeric(df["pred_n"], errors="coerce") - pd.to_numeric(df["gold_n"], errors="coerce")
        print("\n[count delta]")
        print(df["n_delta"].describe().to_string())

    # first diff index 분포(상위)
    fd = df["first_diff_i"].dropna().astype(int)
    if len(fd) > 0:
        print("\n[first_diff_i top]")
        for k, v in fd.value_counts().head(10).items():
            print(f"- i={int(k)}: {int(v)}")

    # 예시: 먼저 count, 그다음 content
    top = int(args.top)
    show_cols = [c for c in ["book_name", "문단식별자", "type", "pred_n", "gold_n", "first_diff_i", "pred_raw", "gold_raw"] if c in df.columns]

    print("\n[examples: count]")
    ex = df[df["type"] == "count"].head(top)
    if len(ex) == 0:
        print("(none)")
    else:
        print(ex[show_cols].to_string(index=False))

    print("\n[examples: content]")
    ex = df[df["type"] == "content"].head(top)
    if len(ex) == 0:
        print("(none)")
    else:
        print(ex[show_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
