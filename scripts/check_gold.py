#!/usr/bin/env python3
"""Gold 데이터 상세 확인"""

import pandas as pd

gold_df = pd.read_csv('datasets/sa/test.csv')
input_df = pd.read_csv('datasets/pa/test.csv')

sent_id = 76
input_row = input_df[input_df['문장식별자'] == sent_id]
gold_rows = gold_df[gold_df['문장식별자'] == sent_id]

print('INPUT TEXT:')
print(repr(str(input_row['번역문'].iloc[0])))
print(f'Length: {len(str(input_row["번역문"].iloc[0]))}')
print()
print(f'GOLD SEGMENTS: {len(gold_rows)} rows')
for i, (_, r) in enumerate(gold_rows.head(8).iterrows()):
    seg_text = str(r['번역문'])
    print(f'  {r["구식별자"]}: len={len(seg_text)}, repr={repr(seg_text)[:50]}...')
