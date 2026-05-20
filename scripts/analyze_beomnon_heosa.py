#!/usr/bin/env python3
"""Analyze heosa co-occurrence for 汎論以斷 target endings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import chi2_contingency


HEOSA = ["夫", "凡", "蓋", "大抵"]
TARGET_CATEGORY = "汎論以斷"
DEFAULT_INPUT = Path("data/sentence_normalized.csv")
DEFAULT_OUTPUT = Path("results/beomnon_heosa_stats.json")


def find_heosa(text: object) -> list[str]:
    if not isinstance(text, str):
        return []
    return [term for term in HEOSA if term in text]


def build_target_mask(df: pd.DataFrame, marker: str | None) -> pd.Series:
    category_mask = df["dansa_category"].eq(TARGET_CATEGORY)
    if marker is None:
        return category_mask
    return category_mask & df["marker_normalized"].eq(marker)


def pct(part: int, whole: int) -> float:
    return round(part / whole * 100, 1) if whole else 0.0


def build_stats(df: pd.DataFrame, marker: str | None) -> dict[str, Any]:
    target_mask = build_target_mask(df, marker)
    target = df[target_mask].copy()
    target["heosa_found"] = target["원문"].apply(find_heosa)
    target["has_heosa"] = target["heosa_found"].apply(bool)

    heosa_counts = {
        term: int(target["원문"].astype(str).str.contains(term, regex=False).sum())
        for term in HEOSA
    }
    has_any_heosa = df["원문"].apply(lambda value: bool(find_heosa(value)))
    target_has_heosa = target["has_heosa"]

    a = int((has_any_heosa & target_mask).sum())
    b = int((has_any_heosa & ~target_mask).sum())
    c = int((~has_any_heosa & target_mask).sum())
    d = int((~has_any_heosa & ~target_mask).sum())
    chi2, p_value, dof, expected = chi2_contingency([[a, b], [c, d]])

    term_corpus = {}
    for term in HEOSA:
        term_mask = df["원문"].astype(str).str.contains(term, regex=False)
        total = int(term_mask.sum())
        in_target = int((term_mask & target_mask).sum())
        term_corpus[term] = {
            "corpus_n": total,
            "target_n": in_target,
            "target_pct": pct(in_target, total),
        }

    target_n = int(len(target))
    target_heosa_n = int(target_has_heosa.sum())
    target_no_heosa_n = target_n - target_heosa_n

    return {
        "input": str(DEFAULT_INPUT),
        "target_category": TARGET_CATEGORY,
        "target_marker": marker or "__category_all__",
        "target_label": (
            f"{TARGET_CATEGORY}/{marker}" if marker else f"{TARGET_CATEGORY} 전체"
        ),
        "matching_rule": "literal substring containment in 원문",
        "target_n": target_n,
        "target_heosa_n": target_heosa_n,
        "target_heosa_pct": pct(target_heosa_n, target_n),
        "target_no_heosa_n": target_no_heosa_n,
        "target_no_heosa_pct": pct(target_no_heosa_n, target_n),
        "heosa_counts_in_target": heosa_counts,
        "heosa_corpus_counts": term_corpus,
        "contingency": {
            "heosa_and_target": a,
            "heosa_and_other": b,
            "no_heosa_and_target": c,
            "no_heosa_and_other": d,
            "chi2": round(float(chi2), 3),
            "p_value": float(p_value),
            "dof": int(dof),
            "expected": [[round(float(v), 3) for v in row] for row in expected],
            "p_target_given_heosa_pct": pct(a, a + b),
            "p_target_given_no_heosa_pct": pct(c, c + d),
            "p_heosa_given_target_pct": pct(a, a + c),
            "p_heosa_given_other_pct": pct(b, b + d),
        },
    }


def print_stats(stats: dict[str, Any]) -> None:
    print("=" * 60)
    print(f"대상: {stats['target_label']}")
    print(f"기준: {stats['matching_rule']}")
    print("=" * 60)
    print(f"  전체: {stats['target_n']}건")
    print(
        "  허사 있음: "
        f"{stats['target_heosa_n']}건 ({stats['target_heosa_pct']:.1f}%)"
    )
    print(
        "  허사 없음: "
        f"{stats['target_no_heosa_n']}건 ({stats['target_no_heosa_pct']:.1f}%)"
    )

    print("\n허사별 대상 내부 출현")
    for term, count in stats["heosa_counts_in_target"].items():
        print(f"  {term}: {count}건")

    print("\n허사별 코퍼스 전체 출현과 대상 포함")
    for term, info in stats["heosa_corpus_counts"].items():
        print(
            f"  {term}: 전체 {info['corpus_n']}건 -> 대상 "
            f"{info['target_n']}건 ({info['target_pct']:.1f}%)"
        )

    ctab = stats["contingency"]
    print("\n2x2 교차표")
    print("                  대상    기타       합계")
    print(
        f"  허사 있음       {ctab['heosa_and_target']:>5,} "
        f"{ctab['heosa_and_other']:>7,} "
        f"{ctab['heosa_and_target'] + ctab['heosa_and_other']:>8,}"
    )
    print(
        f"  허사 없음       {ctab['no_heosa_and_target']:>5,} "
        f"{ctab['no_heosa_and_other']:>7,} "
        f"{ctab['no_heosa_and_target'] + ctab['no_heosa_and_other']:>8,}"
    )
    print(
        f"  합계            {ctab['heosa_and_target'] + ctab['no_heosa_and_target']:>5,} "
        f"{ctab['heosa_and_other'] + ctab['no_heosa_and_other']:>7,} "
        f"{stats['target_n'] + ctab['heosa_and_other'] + ctab['no_heosa_and_other']:>8,}"
    )
    print(f"\n  chi2 = {ctab['chi2']:.3f}, p = {ctab['p_value']:.3e}")
    print(f"  P(대상|허사) = {ctab['p_target_given_heosa_pct']:.1f}%")
    print(f"  P(대상|허사없음) = {ctab['p_target_given_no_heosa_pct']:.1f}%")
    print(f"  P(허사|대상) = {ctab['p_heosa_given_target_pct']:.1f}%")
    print(f"  P(허사|기타) = {ctab['p_heosa_given_other_pct']:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze 夫/凡/蓋/大抵 co-occurrence for 汎論以斷 endings."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--marker",
        default="하나니라",
        help="marker_normalized value to analyze; use --category-all to ignore marker.",
    )
    parser.add_argument(
        "--category-all",
        action="store_true",
        help="Analyze all 汎論以斷 rows, including 하나니 and 하나니라.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print statistics without writing the JSON result.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    marker = None if args.category_all else args.marker
    stats = build_stats(df, marker)
    stats["input"] = str(args.input)

    print_stats(stats)

    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
