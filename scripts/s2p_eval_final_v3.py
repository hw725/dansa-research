#!/usr/bin/env python3
"""
S2P 평가를 위한 문장ID 매핑 및 평가 (최종)
- p2s_full_fixed.xlsx와 datasets/sentence/test.csv는 같은 데이터
- 문장ID만 다름: p2s는 1,2,3..., sent는 원본 ID (76,77,78...)
- 행 순서가 같으므로 행 인덱스로 매핑
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
import json
import re

def norm(t):
    if pd.isna(t): return ''
    return re.sub(r'[\s\t\n\r]+', '', str(t).strip())

def sim(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return SequenceMatcher(None, a, b).ratio()

print("📂 데이터 로딩...")
p2s = pd.read_excel('test_results/p2s_full_fixed.xlsx')
sent = pd.read_csv('datasets/sentence/test.csv')
s2p = pd.read_csv('test_results/s2p_merged_output.csv')
gold = pd.read_csv('datasets/phrase/test_gold_v2.csv')

print(f"  p2s: {len(p2s)} 행")
print(f"  sent: {len(sent)} 행")
print(f"  s2p: {len(s2p)} 행")
print(f"  gold: {len(gold)} 행")

# 1. p2s 문장ID → 원본 문장ID 매핑 (행 순서로)
print("\n🔗 문장ID 매핑 생성...")
p2s_to_orig = {}
for idx in range(len(p2s)):
    p2s_sid = int(p2s.iloc[idx]['문장식별자'])
    orig_sid = int(sent.iloc[idx]['문장식별자'])
    p2s_to_orig[p2s_sid] = orig_sid

print(f"  매핑 수: {len(p2s_to_orig)}")

# 2. S2P 출력에 원본 문장ID 추가
s2p['원본문장식별자'] = s2p['문장식별자'].map(p2s_to_orig)
s2p_valid = s2p.dropna(subset=['원본문장식별자']).copy()
s2p_valid['원본문장식별자'] = s2p_valid['원본문장식별자'].astype(int)

print(f"  S2P 유효 행: {len(s2p_valid)} / {len(s2p)}")

# 저장
s2p_valid.to_csv('test_results/s2p_final_mapped.csv', index=False)
print(f"  저장됨: test_results/s2p_final_mapped.csv")

# 3. 평가
print("\n📊 평가 시작...")

# 문장별 그룹화
gold_map = defaultdict(list)
for _, r in gold.iterrows():
    sid = int(r['문장식별자'])
    gold_map[sid].append({'src': norm(r['원문']), 'tgt': norm(r['번역문'])})

pred_map = defaultdict(list)
for _, r in s2p_valid.iterrows():
    sid = int(r['원본문장식별자'])
    pred_map[sid].append({'src': norm(r['원문']), 'tgt': norm(r['번역문'])})

# 공통 문장
common = set(gold_map.keys()) & set(pred_map.keys())
print(f"  Gold 문장: {len(gold_map)}")
print(f"  Pred 문장: {len(pred_map)}")
print(f"  공통 문장: {len(common)} ({100*len(common)/len(gold_map):.1f}% coverage)")

# 평가
total_tp, total_fp, total_fn = 0, 0, 0
all_sims = []
seg_matches = 0
src_exact = 0

for sid in common:
    g_segs = gold_map[sid]
    p_segs = pred_map[sid]
    
    # 원문 무결성
    g_src = ''.join([s['src'] for s in g_segs])
    p_src = ''.join([s['src'] for s in p_segs])
    if g_src == p_src:
        src_exact += 1
    
    # 구 개수
    if len(g_segs) == len(p_segs):
        seg_matches += 1
    
    # 번역문 F1
    g_tgts = [s['tgt'] for s in g_segs]
    p_tgts = [s['tgt'] for s in p_segs]
    
    matched = set()
    for pt in p_tgts:
        best_sim = 0
        best_i = -1
        for i, gt in enumerate(g_tgts):
            if i in matched: continue
            s = sim(pt, gt)
            if s > best_sim:
                best_sim = s
                best_i = i
        
        all_sims.append(best_sim)
        if best_sim >= 0.8 and best_i >= 0:
            matched.add(best_i)
            total_tp += 1
        else:
            total_fp += 1
    
    total_fn += len(g_tgts) - len(matched)

# 계산
prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
high_sim = sum(1 for s in all_sims if s >= 0.9)

# 결과
print("\n" + "="*60)
print("📊 S2P 최종 평가 결과")
print("="*60)

print(f"\n📌 커버리지:")
print(f"  매칭 문장: {len(common)} / {len(gold_map)} ({100*len(common)/len(gold_map):.1f}%)")
print(f"  원문 Exact Match: {100*src_exact/len(common):.1f}%")

print(f"\n📌 구 정확도:")
print(f"  구 개수 일치율: {100*seg_matches/len(common):.1f}%")

print(f"\n📌 ⭐ 구 단위 F1:")
print(f"  Precision: {100*prec:.2f}%")
print(f"  Recall: {100*rec:.2f}%")
print(f"  F1 Score: {100*f1:.2f}% ⬅️ 핵심 지표")
print(f"  평균 유사도: {100*np.mean(all_sims):.2f}%")
print(f"  High Sim (≥0.9): {100*high_sim/len(all_sims):.1f}%")

print("="*60)

# 저장
results = {
    'coverage': len(common) / len(gold_map),
    'matched_sentences': len(common),
    'src_exact_match': src_exact / len(common) if common else 0,
    'seg_count_match': seg_matches / len(common) if common else 0,
    'precision': prec,
    'recall': rec,
    'f1': f1,
    'avg_similarity': float(np.mean(all_sims)) if all_sims else 0,
    'high_sim_rate': high_sim / len(all_sims) if all_sims else 0
}

with open('test_results/s2p_eval_final.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n💾 저장됨: test_results/s2p_eval_final.json")
