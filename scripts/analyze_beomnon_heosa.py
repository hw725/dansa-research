#!/usr/bin/env python3
"""汎論以斷 허사 양방향 분석

방향1: 하나니(라) → 허사 동반율
방향2: 허사 포함 문장 → 하나니(라) 종결 비율
"""
import pandas as pd

df = pd.read_csv('data/sentence_normalized.csv', encoding='utf-8')
HEOSA = ['夫', '凡', '蓋', '大抵']

beomnon = df[df['dansa_category'] == '汎論以斷'].copy()

# ============================================================
# 방향1: 하나니(라) → 허사 동반율
# ============================================================
print("=" * 60)
print("방향1: 하나니(라) 문장 중 허사 동반율")
print("=" * 60)

def find_heosa(text):
    if not isinstance(text, str):
        return []
    return [h for h in HEOSA if h in text]

beomnon['heosa_found'] = beomnon['원문'].apply(find_heosa)
beomnon['has_heosa'] = beomnon['heosa_found'].apply(lambda x: len(x) > 0)

total_b = len(beomnon)
with_h = beomnon['has_heosa'].sum()
print(f"  전체: {total_b}건")
print(f"  허사 있음: {with_h}건 ({with_h/total_b*100:.1f}%)")
print(f"  허사 없음: {total_b - with_h}건 ({(total_b - with_h)/total_b*100:.1f}%)")

# ============================================================
# 방향2: 허사 포함 문장 → 종결어미 분포
# ============================================================
print(f"\n{'=' * 60}")
print("방향2: 허사 포함 문장의 종결어미 분포")
print("=" * 60)

for h in HEOSA:
    mask = df['원문'].apply(lambda x: h in str(x))
    h_sentences = df[mask]
    total_h = len(h_sentences)
    if total_h == 0:
        print(f"\n{h}: 0건")
        continue

    cats = h_sentences['dansa_category'].value_counts()
    beomnon_cnt = cats.get('汎論以斷', 0)
    print(f"\n{h}: 전체 {total_h}건 → 汎論以斷 {beomnon_cnt}건 ({beomnon_cnt/total_h*100:.1f}%)")
    print("  종결어미 분포:")
    for cat, cnt in cats.items():
        if cat:
            print(f"    {cat}: {cnt} ({cnt/total_h*100:.1f}%)")
    empty = (h_sentences['dansa_category'] == '').sum()
    if empty:
        print(f"    (미분류): {empty} ({empty/total_h*100:.1f}%)")

# 전체 허사 통합
print(f"\n{'=' * 60}")
print("전체 허사 통합")
print("=" * 60)

any_heosa = df['원문'].apply(lambda x: any(h in str(x) for h in HEOSA))
heosa_all = df[any_heosa]
total_ha = len(heosa_all)
cats_all = heosa_all['dansa_category'].value_counts()
b_cnt = cats_all.get('汎論以斷', 0)

print(f"허사(夫/凡/蓋/大抵) 포함 문장: {total_ha}건")
print(f"  → 汎論以斷: {b_cnt}건 ({b_cnt/total_ha*100:.1f}%)")
print(f"  → 기타 종결: {total_ha - b_cnt}건 ({(total_ha - b_cnt)/total_ha*100:.1f}%)")
print("\n  종결어미 분포:")
for cat, cnt in cats_all.items():
    if cat:
        print(f"    {cat}: {cnt} ({cnt/total_ha*100:.1f}%)")
empty_all = (heosa_all['dansa_category'] == '').sum()
if empty_all:
    print(f"    (미분류): {empty_all} ({empty_all/total_ha*100:.1f}%)")

# ============================================================
# 요약 2x2 테이블
# ============================================================
print(f"\n{'=' * 60}")
print("2x2 교차표")
print("=" * 60)

has_heosa_col = df['원문'].apply(lambda x: any(h in str(x) for h in HEOSA))
is_beomnon_col = df['dansa_category'] == '汎論以斷'

a = (has_heosa_col & is_beomnon_col).sum()   # 허사O, 汎論O
b = (has_heosa_col & ~is_beomnon_col).sum()  # 허사O, 汎論X
c = (~has_heosa_col & is_beomnon_col).sum()  # 허사X, 汎論O
d = (~has_heosa_col & ~is_beomnon_col).sum() # 허사X, 汎論X

print(f"                  汎論以斷    기타      합계")
print(f"  허사 있음       {a:>7,}    {b:>7,}    {a+b:>7,}")
print(f"  허사 없음       {c:>7,}    {d:>7,}    {c+d:>7,}")
print(f"  합계            {a+c:>7,}    {b+d:>7,}    {a+b+c+d:>7,}")

from scipy.stats import chi2_contingency, fisher_exact
ct = [[a, b], [c, d]]
chi2, pv, dof, expected = chi2_contingency(ct)
odds_ratio = (a * d) / (b * c) if b * c > 0 else float('inf')
print(f"\n  chi2 = {chi2:.1f}, p = {pv:.2e}")
print(f"  odds ratio = {odds_ratio:.1f}")
print(f"  P(汎論|허사) = {a/(a+b)*100:.1f}%")
print(f"  P(汎論|허사없음) = {c/(c+d)*100:.2f}%")
print(f"  P(허사|汎論) = {a/(a+c)*100:.1f}%")
print(f"  P(허사|汎論아님) = {b/(b+d)*100:.1f}%")
