#!/usr/bin/env python3
"""
S2P 입력/Gold 데이터 매칭 분석
"""
import pandas as pd
import re

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s\t\n\r]+', '', str(t).strip())

# 데이터 로드
sent_test = pd.read_csv('datasets/sentence/test.csv')
p2s_full = pd.read_excel('test_results/p2s_full_fixed.xlsx')
phrase_gold = pd.read_csv('datasets/phrase/test.csv')

print('=== 데이터 구조 ===')
print('sent_test columns:', sent_test.columns.tolist())
print('p2s_full columns:', p2s_full.columns.tolist())
print('phrase_gold columns:', phrase_gold.columns.tolist())

# 원문 정규화
sent_test['src_norm'] = sent_test['원문'].apply(norm)
p2s_full['src_norm'] = p2s_full['원문'].apply(norm)

# sent_test의 원문 집합
sent_src_set = set(sent_test['src_norm'].values)
print(f'\n=== 매칭 분석 ===')
print(f'sent_test 고유 원문: {len(sent_src_set)}')

# p2s_full에서 매칭
p2s_matched = p2s_full[p2s_full['src_norm'].isin(sent_src_set)]
print(f'p2s_full에서 매칭: {len(p2s_matched)} rows')
print(f'p2s_full 고유 문장ID 중 매칭: {p2s_matched["문장식별자"].nunique()}')

# S2P 출력에서 해당 문장 추출
s2p_merged = pd.read_csv('test_results/s2p_merged_output.csv')
s2p_matched_ids = set(p2s_matched['문장식별자'].values)
s2p_filtered = s2p_merged[s2p_merged['문장식별자'].isin(s2p_matched_ids)]

print(f'\nS2P 출력에서 매칭된 문장: {s2p_filtered["문장식별자"].nunique()}')
print(f'S2P 출력 필터링 후 행 수: {len(s2p_filtered)}')

# 필터링된 데이터 저장
s2p_filtered.to_csv('test_results/s2p_test_subset.csv', index=False)
print(f'\n저장됨: test_results/s2p_test_subset.csv')

# sent_test → 문장ID 매핑 생성
sent_src_to_orig_id = {}
for _, r in sent_test.iterrows():
    src = norm(r['원문'])
    orig_id = int(r['문장식별자'])
    if src not in sent_src_to_orig_id:
        sent_src_to_orig_id[src] = orig_id

# p2s_full 문장ID → 원본 문장ID 매핑
p2s_id_to_orig_id = {}
for _, r in p2s_full.iterrows():
    src = r['src_norm']
    p2s_id = int(r['문장식별자'])
    if src in sent_src_to_orig_id:
        p2s_id_to_orig_id[p2s_id] = sent_src_to_orig_id[src]

print(f'\n문장ID 매핑 생성: {len(p2s_id_to_orig_id)} 쌍')

# S2P 출력에 원본 문장ID 추가
s2p_filtered['원본문장식별자'] = s2p_filtered['문장식별자'].map(p2s_id_to_orig_id)
s2p_filtered_valid = s2p_filtered.dropna(subset=['원본문장식별자'])
s2p_filtered_valid['원본문장식별자'] = s2p_filtered_valid['원본문장식별자'].astype(int)

print(f'원본 문장ID 매핑 후 유효 행: {len(s2p_filtered_valid)}')

# 원본 문장ID로 저장
s2p_filtered_valid.to_csv('test_results/s2p_test_mapped.csv', index=False)
print(f'저장됨: test_results/s2p_test_mapped.csv')
