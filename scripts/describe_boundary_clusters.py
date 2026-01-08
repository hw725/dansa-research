#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import regex as re


@dataclass
class LabelGuess:
    label: str
    reason: str
    score: float


_RE_ATTR = re.compile(r"^(?:子曰|有子曰|孟子曰|程子曰|范氏曰|何氏曰|謝氏曰|史記世家曰|又曰)$")
_RE_SAYS = re.compile(r"曰$")
_RE_GLOSS = re.compile(r"(?:^|\s)(?:.*?)(?:者는|者는|者는|者는|者는|者는)\s*")  # weak, kept for compatibility
_RE_DEFINE = re.compile(r"^[^\s]{1,10}(?:은|는)\s+.+(?:也|矣|라)[^\s]*$")
_RE_YEAR = re.compile(r"[一二三四五六七八九十百千0-9]{1,4}年")
_RE_TIMESEQ = re.compile(r"^(?:\s*)?(?:又|故|乃|遂|及|則|然|若|其)\b")


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s == "nan" else s


def guess_label(df: pd.DataFrame) -> LabelGuess:
    left = df["src_left"].astype(str).map(_safe_str)
    right = df["src_right"].astype(str).map(_safe_str)

    n = max(1, len(df))

    frac_attr = float(left.map(lambda s: bool(_RE_ATTR.match(s.strip()))).sum()) / n
    frac_says = float(left.map(lambda s: bool(_RE_SAYS.search(s.strip()))).sum()) / n
    frac_define = float(left.map(lambda s: bool(_RE_DEFINE.match(s.strip()))).sum()) / n
    frac_year = float((left + " " + right).map(lambda s: bool(_RE_YEAR.search(s))).sum()) / n

    # Simple rule-based guesses
    candidates: list[LabelGuess] = []

    if frac_attr >= 0.35 or frac_says >= 0.55:
        candidates.append(
            LabelGuess(
                label="발화/인용 도입(曰/…曰)",
                reason=f"src_left에 '曰' 패턴 다수 (attr={frac_attr:.2f}, says={frac_says:.2f})",
                score=max(frac_attr, frac_says),
            )
        )

    if frac_define >= 0.25:
        candidates.append(
            LabelGuess(
                label="용어 풀이/주석 정의(…은/…는 …也/…라)",
                reason=f"정의문 형태 다수 (define={frac_define:.2f})",
                score=frac_define,
            )
        )

    if frac_year >= 0.15:
        candidates.append(
            LabelGuess(
                label="연표/서술(연도·사건 나열)",
                reason=f"연도 표기 포함 비율 높음 (year={frac_year:.2f})",
                score=frac_year,
            )
        )

    # fallback label
    if not candidates:
        # try to detect "continuation/segmentation" via short-left / long-right or connective cues
        left_len = left.map(len)
        right_len = right.map(len)
        frac_short_left = float((left_len <= 6).sum()) / n
        frac_long_right = float((right_len >= 18).sum()) / n
        if frac_short_left >= 0.35 and frac_long_right >= 0.35:
            return LabelGuess(
                label="짧은 도입부 → 본문 전개(접속/요약)",
                reason=f"짧은 left + 긴 right 비율 (short_left={frac_short_left:.2f}, long_right={frac_long_right:.2f})",
                score=min(frac_short_left, frac_long_right),
            )
        return LabelGuess(label="기타/혼합(추가 해석 필요)", reason="명확한 규칙 패턴 부족", score=0.0)

    candidates.sort(key=lambda c: (-c.score, c.label))
    return candidates[0]


def format_examples(df: pd.DataFrame, k: int) -> str:
    k = int(k)
    rows = df.head(k)
    lines: list[str] = []
    for _, r in rows.iterrows():
        lines.append(f"- book={r.get('book_name','')}, para={r.get('paragraph_id','')}")
        lines.append(f"  - src_L: {_safe_str(r.get('src_left'))}")
        lines.append(f"  - src_R: {_safe_str(r.get('src_right'))}")
        tgt_l = _safe_str(r.get('tgt_left'))
        tgt_r = _safe_str(r.get('tgt_right'))
        if tgt_l or tgt_r:
            lines.append(f"  - tgt_L: {tgt_l}")
            lines.append(f"  - tgt_R: {tgt_r}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Heuristic natural-language descriptions for boundary clusters")
    ap.add_argument("--csv", type=Path, default=Path("hyeonto/reports/boundary_function_clusters/boundary_clusters.csv"))
    ap.add_argument("--out", type=Path, default=Path("hyeonto/reports/boundary_function_clusters/boundary_clusters_labeled.md"))
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "cluster_id" not in df.columns:
        raise SystemExit(f"cluster_id 컬럼이 없습니다: {sorted(df.columns)}")

    out_lines: list[str] = []
    out_lines.append("# boundary clusters (heuristic labels)\n")
    out_lines.append(f"- source: {args.csv}")
    out_lines.append("- note: 라벨은 휴리스틱 추정이며, 대표 예문 기반으로 검토/수정 가능\n")

    for cid, g in df.groupby("cluster_id"):
        g2 = g.copy()
        # stable, readable samples
        g2 = g2.sort_values(["book_name", "paragraph_id", "left_sentence_id"], kind="mergesort")
        guess = guess_label(g2)

        out_lines.append(f"## cluster {int(cid)} (n={len(g2)})")
        out_lines.append(f"- guess: **{guess.label}** (score={guess.score:.2f})")
        out_lines.append(f"- reason: {guess.reason}")
        out_lines.append("")
        out_lines.append(format_examples(g2, k=int(args.examples)))
        out_lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
