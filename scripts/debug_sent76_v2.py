#!/usr/bin/env python3
"""문장 76 상세 비교"""
import pandas as pd
import re

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s]+', '', str(t).strip())

gold = pd.read_csv('datasets/phrase/test_gold_v2.csv')
sent = pd.read_csv('datasets/sentence/test.csv')

# 문장 76 상세 비교
sid = 76
g_rows = gold[gold['문장식별자'] == sid]
s_rows = sent[sent['문장식별자'] == sid]

print(f'sentence test (문장 76 - {len(s_rows)} rows):')
for _, r in s_rows.iterrows():
    src = norm(r["원문"])
    print(f'  [{src}]')
    
print(f'\nGold v2 (문장 76 - {len(g_rows)} rows):')
for i, (_, r) in enumerate(g_rows.iterrows()):
    if i >= 10:
        print(f'  ... 외 {len(g_rows) - 10}개')
        break
    src = norm(r["원문"])
    print(f'  [{src}]')

# 합계 비교
s_full = ''.join([norm(r["원문"]) for _, r in s_rows.iterrows()])
g_full = ''.join([norm(r["원문"]) for _, r in g_rows.iterrows()])

print(f'\n합계:')
print(f'  sent ({len(s_full)} chars): [{s_full[:80]}]')
print(f'  gold ({len(g_full)} chars): [{g_full[:80]}]')
print(f'  일치: {s_full == g_full}')
print(f'  sent가 gold에 포함: {s_full in g_full}')
