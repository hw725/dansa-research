#!/usr/bin/env python3
"""斷辭 카테고리별 예시 샘플러 — 원문(한문) + 번역문 (논문 인용용).

각 카테고리에서 무작위 N건을 추출해 book·marker·원문·번역문을 출력한다.
汎論以斷의 경우 --heosa 옵션으로 夫·凡·蓋·大抵 동반/비동반을 분리 표시.

Usage:
    python scripts/sample_dansa_examples.py                       # 6 카테고리 모두 각 5건
    python scripts/sample_dansa_examples.py 微絶之斷               # 특정 카테고리
    python scripts/sample_dansa_examples.py 汎論以斷 --heosa       # 허사 동반/비동반 분리
    python scripts/sample_dansa_examples.py --n 10                # 카테고리당 10건
    python scripts/sample_dansa_examples.py --seed 7              # 시드 변경
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CATEGORIES = ['游辭以斷', '夬絶之斷', '微絶之斷', '記史之斷', '敍述之斷', '汎論以斷']
HEOSA = ['夫', '凡', '蓋', '大抵']
DATA = Path('data/sentence_normalized.csv')


def print_example(r: pd.Series, src_max: int = 200, tgt_max: int = 200) -> None:
    book = r.get('book', '')
    mk = r.get('marker_normalized', '')
    src = str(r.get('원문', ''))[:src_max]
    tgt = str(r.get('번역문', ''))[:tgt_max]
    print(f"\n  [{book}] marker: {mk}")
    print(f"    원문: {src}")
    print(f"    번역: {tgt}")


def main() -> int:
    parser = argparse.ArgumentParser(description='斷辭 예시 샘플러')
    parser.add_argument('category', nargs='?', choices=CATEGORIES,
                        help='특정 카테고리만 샘플링 (생략 시 전체)')
    parser.add_argument('-n', type=int, default=5, help='카테고리당 샘플 수 (기본 5)')
    parser.add_argument('--heosa', action='store_true',
                        help='汎論以斷에서 허사(夫·凡·蓋·大抵) 동반/비동반 분리 출력')
    parser.add_argument('--seed', type=int, default=42, help='샘플링 random_state')
    args = parser.parse_args()

    if not DATA.exists():
        print(f"ERROR: {DATA} 부재. prepare_sentence_dataset.py 먼저 실행.", file=sys.stderr)
        return 1

    df = pd.read_csv(DATA, encoding='utf-8')
    cats = [args.category] if args.category else CATEGORIES

    for cat in cats:
        sub = df[df['dansa_category'] == cat]
        if sub.empty:
            print(f"\n[skip] {cat}: 0건")
            continue

        print(f"\n{'='*72}")
        print(f"{cat} — 전체 {len(sub):,}건 (시드={args.seed})")
        print(f"{'='*72}")

        if args.heosa and cat == '汎論以斷':
            sub = sub.copy()
            sub['_has_heosa'] = sub['원문'].apply(
                lambda x: any(h in str(x) for h in HEOSA)
            )
            for label, mask in (('허사 동반', sub['_has_heosa']),
                                ('허사 미동반', ~sub['_has_heosa'])):
                pool = sub[mask]
                if pool.empty:
                    continue
                bucket = pool.sample(min(args.n, len(pool)), random_state=args.seed)
                print(f"\n--- {label} ({len(pool):,}건 중 {len(bucket)}건) ---")
                for _, r in bucket.iterrows():
                    print_example(r)
        else:
            samples = sub.sample(min(args.n, len(sub)), random_state=args.seed)
            for _, r in samples.iterrows():
                print_example(r)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
