#!/usr/bin/env python3
"""S2P 입출력 원문 무결성 검증 v2 - sent_test도 문장식별자로 그룹화"""
import pandas as pd
import re
from collections import defaultdict

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s\t\n\r]+', '', str(t).strip())

# S2P 출력 (매핑된 원본 문장식별자 포함)
s2p_out = pd.read_csv('test_results/s2p_test_mapped.csv')
sent_test = pd.read_csv('datasets/sentence/test.csv')

print('=== S2P 입출력 원문 무결성 검증 v2 ===')

# sent_test를 문장식별자별로 그룹화
sent_map = defaultdict(str)
for _, r in sent_test.iterrows():
    sid = int(r['문장식별자'])
    sent_map[sid] += norm(r['원문'])

# S2P 출력을 원본문장식별자별로 그룹화
s2p_map = defaultdict(str)
for _, r in s2p_out.iterrows():
    orig_sid = int(r['원본문장식별자'])
    s2p_map[orig_sid] += norm(r['원문'])

# 비교
common_sids = set(sent_map.keys()) & set(s2p_map.keys())
print(f'공통 문장: {len(common_sids)}')

matches = 0
mismatches = []

for sid in common_sids:
    in_src = sent_map[sid]
    out_src = s2p_map[sid]
    
    if in_src == out_src:
        matches += 1
    else:
        mismatches.append({
            'sid': sid,
            'in_len': len(in_src),
            'out_len': len(out_src),
            'in_src': in_src[:50],
            'out_src': out_src[:50]
        })

print(f'일치: {matches} / {len(common_sids)} ({100*matches/len(common_sids):.1f}%)')
print(f'불일치: {len(mismatches)}')

if mismatches:
    print('\n샘플 불일치:')
    for m in mismatches[:5]:
        print(f'  문장 {m["sid"]}: 입력 {m["in_len"]} → 출력 {m["out_len"]} chars')
        print(f'    입력: [{m["in_src"]}...]')
        print(f'    출력: [{m["out_src"]}...]')
