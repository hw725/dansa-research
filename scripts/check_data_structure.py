#!/usr/bin/env python3
"""데이터 구조 디버깅"""

import pandas as pd

print("=== PA/TEST.CSV ===")
input_df = pd.read_csv('datasets/p2s/test.csv')
print(f"Columns: {list(input_df.columns)}")
print("First 3 rows:")
print(input_df[['문장식별자', '원문', '번역문']].head(3))
print()

print("=== SA/TEST.CSV ===")
gold_df = pd.read_csv('datasets/s2p/test.csv')
print(f"Columns: {list(gold_df.columns)}")
print("First 3 rows:")
print(gold_df[['문장식별자', '구식별자', '원문', '번역문']].head(3))
print()

# 문장 76번 상세
print("=== 문장 76번 비교 ===")
input_row = input_df[input_df['문장식별자'] == 76]
gold_rows = gold_df[gold_df['문장식별자'] == 76]

print(f"INPUT (PA): {len(input_row)} row")
if not input_row.empty:
    src = str(input_row['원문'].iloc[0])
    tgt = str(input_row['번역문'].iloc[0])
    print(f"  원문 len={len(src)}: {src[:80]}...")
    print(f"  번역문 len={len(tgt)}: {tgt[:80]}...")

print(f"\nGOLD (SA): {len(gold_rows)} rows")
# 모든 구의 번역문을 연결
all_tgt = ''.join([str(r) for r in gold_rows['번역문']])
print(f"  연결된 번역문 len={len(all_tgt)}: {all_tgt[:80]}...")

# 원문 연결
all_src = ''.join([str(r) for r in gold_rows['원문']])
print(f"  연결된 원문 len={len(all_src)}: {all_src[:80]}...")
