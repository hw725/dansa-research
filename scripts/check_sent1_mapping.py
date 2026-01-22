#!/usr/bin/env python3
"""문장 1 매핑 상세 확인"""
import pandas as pd
import re

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s]+', '', str(t).strip())

sent_test = pd.read_csv('datasets/sentence/test.csv')
s2p_mapped = pd.read_csv('test_results/s2p_test_mapped.csv')

# sent_test 문장 1
sent1 = sent_test[sent_test['문장식별자'] == 1]
print('sent_test 문장식별자=1:')
sent1_full = ''
for _, r in sent1.iterrows():
    s = norm(r["원문"])
    sent1_full += s
    print(f'  [{s[:40]}...]')
print(f'  합계: [{sent1_full[:60]}...]')

# s2p_mapped에서 원본문장식별자=1인 것
s2p_orig1 = s2p_mapped[s2p_mapped['원본문장식별자'] == 1]
print(f'\ns2p_mapped 원본문장식별자=1 ({len(s2p_orig1)} rows):')
s2p1_full = ''
if len(s2p_orig1) > 0:
    for _, r in s2p_orig1.iterrows():
        s = norm(r["원문"])
        s2p1_full += s
        print(f'  [{s[:40]}...]')
    print(f'  합계: [{s2p1_full[:60]}...]')

print(f'\n일치: {sent1_full == s2p1_full}')
