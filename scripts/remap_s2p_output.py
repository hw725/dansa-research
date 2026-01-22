#!/usr/bin/env python3
"""
S2P 출력에 원본 문장식별자 매핑 (수정 버전)
- sent_test를 문장식별자별로 그룹화하여 전체 원문 생성
- p2s_full의 각 행 원문과 매칭
"""
import pandas as pd
import re
from collections import defaultdict

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s\t\n\r]+', '', str(t).strip())

print("📂 데이터 로딩...")
sent_test = pd.read_csv('datasets/sentence/test.csv')
p2s_full = pd.read_excel('test_results/p2s_full_fixed.xlsx')
s2p_merged = pd.read_csv('test_results/s2p_merged_output.csv')

print(f"  sent_test: {len(sent_test)} 행, {sent_test['문장식별자'].nunique()} 문장")
print(f"  p2s_full: {len(p2s_full)} 행")
print(f"  s2p_merged: {len(s2p_merged)} 행")

# 1. sent_test를 문장식별자별로 그룹화하여 전체 원문 생성
print("\n📊 sent_test 문장별 원문 생성...")
sent_full_src = defaultdict(str)
for _, r in sent_test.iterrows():
    sid = int(r['문장식별자'])
    sent_full_src[sid] += norm(r['원문'])

print(f"  고유 문장: {len(sent_full_src)}")

# 2. 역매핑: 전체 원문 → 원본 문장식별자
src_to_orig_id = {}
for sid, src in sent_full_src.items():
    if src not in src_to_orig_id:
        src_to_orig_id[src] = sid

print(f"  고유 원문: {len(src_to_orig_id)}")

# 3. p2s_full의 각 행 원문 → 원본 문장식별자 매핑
print("\n🔗 p2s_full → 원본 문장ID 매핑...")
p2s_id_to_orig_id = {}
matched = 0
for _, r in p2s_full.iterrows():
    p2s_id = int(r['문장식별자'])
    p2s_src = norm(r['원문'])
    
    if p2s_src in src_to_orig_id:
        p2s_id_to_orig_id[p2s_id] = src_to_orig_id[p2s_src]
        matched += 1

print(f"  매칭: {matched} / {len(p2s_full)} ({100*matched/len(p2s_full):.1f}%)")
print(f"  매핑 쌍: {len(p2s_id_to_orig_id)}")

# 4. S2P 출력에 원본 문장ID 추가
print("\n📝 S2P 출력에 원본 문장ID 추가...")
s2p_merged['원본문장식별자'] = s2p_merged['문장식별자'].map(p2s_id_to_orig_id)

# 유효한 행만 필터링
s2p_valid = s2p_merged.dropna(subset=['원본문장식별자']).copy()
s2p_valid['원본문장식별자'] = s2p_valid['원본문장식별자'].astype(int)

print(f"  유효 행: {len(s2p_valid)} / {len(s2p_merged)}")
print(f"  유효 문장: {s2p_valid['원본문장식별자'].nunique()}")

# 5. 저장
output_path = 'test_results/s2p_test_mapped_v2.csv'
s2p_valid.to_csv(output_path, index=False)
print(f"\n💾 저장됨: {output_path}")

# 6. 검증
print("\n🔍 검증...")
sample_sids = list(s2p_valid['원본문장식별자'].unique())[:5]
for orig_sid in sample_sids:
    # sent_test에서
    sent_src = sent_full_src[orig_sid]
    
    # s2p 출력에서
    s2p_rows = s2p_valid[s2p_valid['원본문장식별자'] == orig_sid]
    s2p_src = ''.join([norm(r['원문']) for _, r in s2p_rows.iterrows()])
    
    match = "✅" if sent_src == s2p_src else "❌"
    print(f"  문장 {orig_sid}: {match} (sent {len(sent_src)} chars, s2p {len(s2p_src)} chars)")
