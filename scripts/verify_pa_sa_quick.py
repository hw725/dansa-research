#!/usr/bin/env python3
"""PA와 SA 텍스트 일치 검증 (간단 버전)"""
import pandas as pd
import re

def normalize_text(text):
    if pd.isna(text):
        return ''
    return re.sub(r'\s+', '', str(text))

pa = pd.read_csv('hyeonto/datasets/pa_train_merged.csv')
sa = pd.read_csv('hyeonto/datasets/sa_train_merged.csv')

# 공통 문장
pa_sents = set(zip(pa['book_name'], pa['문장식별자']))
sa_sents = set(zip(sa['book_name'], sa['문장식별자']))
common = pa_sents & sa_sents

print(f"Total common sentences: {len(common):,}")

# 1000개 샘플 테스트
import random
sample = random.sample(list(common), min(1000, len(common)))

match = 0
mismatch = 0

for book, sent_id in sample:
    pa_row = pa[(pa['book_name'] == book) & (pa['문장식별자'] == sent_id)]
    sa_rows = sa[(sa['book_name'] == book) & (sa['문장식별자'] == sent_id)]

    if len(pa_row) == 0 or len(sa_rows) == 0:
        continue

    pa_norm = normalize_text(pa_row.iloc[0]['원문'])
    sa_norm = normalize_text(''.join(sa_rows['원문'].astype(str).tolist()))

    if pa_norm == sa_norm:
        match += 1
    else:
        mismatch += 1

print(f"\nSample verification (1000 sentences):")
print(f"  Match: {match} ({match/(match+mismatch)*100:.2f}%)")
print(f"  Mismatch: {mismatch} ({mismatch/(match+mismatch)*100:.2f}%)")

if mismatch == 0:
    print("\n[OK] PA and SA texts match perfectly!")
else:
    print(f"\n[WARNING] Found {mismatch} mismatches in sample")
