#!/usr/bin/env python3
"""
S2P 평가 - 원문 텍스트 기반 매칭
문장식별자 체계가 다른 경우를 위해, 원문 전체 텍스트로 매칭
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
import argparse
import json
import re

def normalize_text(text):
    """공백/구두점 제거"""
    if pd.isna(text):
        return ""
    s = str(text).strip()
    s = re.sub(r'[\s\t\n\r]+', '', s)
    return s

def calculate_similarity(text1, text2):
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()

def build_sentence_map(df):
    """문장식별자별 전체 원문과 구 리스트 생성"""
    grouped = defaultdict(lambda: {'src_full': '', 'tgt_full': '', 'segments': []})
    
    for _, row in df.iterrows():
        sent_id = int(row['문장식별자'])
        src = normalize_text(row['원문'])
        tgt = normalize_text(row['번역문'])
        grouped[sent_id]['src_full'] += src
        grouped[sent_id]['tgt_full'] += tgt
        grouped[sent_id]['segments'].append({'src': src, 'tgt': tgt})
    
    return grouped

def match_by_source_text(gold_map, pred_map):
    """원문 텍스트가 동일한 문장끼리 매칭 (완전 일치만)"""
    # Gold 원문 → 문장ID 인덱스
    gold_src_to_id = {}
    for sent_id, data in gold_map.items():
        src_key = data['src_full']
        if src_key not in gold_src_to_id:
            gold_src_to_id[src_key] = sent_id
    
    matched_pairs = []
    unmatched_pred = []
    
    for pred_id, pred_data in pred_map.items():
        pred_src = pred_data['src_full']
        
        if pred_src in gold_src_to_id:
            gold_id = gold_src_to_id[pred_src]
            matched_pairs.append((gold_id, pred_id))
        else:
            unmatched_pred.append(pred_id)
    
    return matched_pairs, unmatched_pred

def evaluate_segments(gold_segs, pred_segs):
    """구 단위 평가"""
    gold_tgts = [s['tgt'] for s in gold_segs]
    pred_tgts = [s['tgt'] for s in pred_segs]
    
    # F1 계산
    tp = 0
    matched_gold = set()
    all_sims = []
    
    for pred_tgt in pred_tgts:
        best_sim = 0
        best_idx = -1
        for i, gold_tgt in enumerate(gold_tgts):
            if i in matched_gold:
                continue
            sim = calculate_similarity(pred_tgt, gold_tgt)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        
        all_sims.append(best_sim)
        if best_sim >= 0.8 and best_idx >= 0:
            matched_gold.add(best_idx)
            tp += 1
    
    fp = len(pred_tgts) - tp
    fn = len(gold_tgts) - len(matched_gold)
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_gold': len(gold_tgts),
        'n_pred': len(pred_tgts),
        'avg_sim': np.mean(all_sims) if all_sims else 0,
        'high_sim': sum(1 for s in all_sims if s >= 0.9)
    }

def main(gold_file, pred_file, output_file):
    print(f"📂 Gold 로딩: {gold_file}")
    gold_df = pd.read_csv(gold_file) if gold_file.endswith('.csv') else pd.read_excel(gold_file)
    
    print(f"📂 Pred 로딩: {pred_file}")
    pred_df = pd.read_csv(pred_file) if pred_file.endswith('.csv') else pd.read_excel(pred_file)
    
    print(f"  Gold: {len(gold_df)} rows, Pred: {len(pred_df)} rows")
    
    # 문장별 그룹화
    gold_map = build_sentence_map(gold_df)
    pred_map = build_sentence_map(pred_df)
    
    print(f"\n📊 문장 수:")
    print(f"  Gold: {len(gold_map)}")
    print(f"  Pred: {len(pred_map)}")
    
    # 원문 텍스트 기반 매칭
    print("\n🔗 원문 텍스트 기반 매칭 중...")
    matched_pairs, unmatched = match_by_source_text(gold_map, pred_map)
    
    print(f"  매칭된 문장 쌍: {len(matched_pairs)}")
    print(f"  미매칭 Pred 문장: {len(unmatched)}")
    print(f"  커버리지: {100*len(matched_pairs)/len(gold_map):.1f}%")
    
    # 구 단위 평가
    print("\n📐 구 단위 F1 계산 중...")
    total_tp, total_fp, total_fn = 0, 0, 0
    all_sims = []
    high_sim_count = 0
    seg_count_matches = 0
    
    for gold_id, pred_id in matched_pairs:
        gold_segs = gold_map[gold_id]['segments']
        pred_segs = pred_map[pred_id]['segments']
        
        result = evaluate_segments(gold_segs, pred_segs)
        total_tp += result['tp']
        total_fp += result['fp']
        total_fn += result['fn']
        all_sims.append(result['avg_sim'])
        high_sim_count += result['high_sim']
        
        if result['n_gold'] == result['n_pred']:
            seg_count_matches += 1
    
    # F1 계산
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    total_pred_segs = total_tp + total_fp
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 S2P 평가 결과 (원문 텍스트 매칭 기반)")
    print("="*60)
    
    print(f"\n📌 커버리지:")
    print(f"  매칭된 문장: {len(matched_pairs)} / {len(gold_map)} ({100*len(matched_pairs)/len(gold_map):.1f}%)")
    
    print(f"\n📌 구 개수 정확도:")
    print(f"  구 개수 일치율: {100*seg_count_matches/len(matched_pairs):.1f}%")
    
    print(f"\n📌 ⭐ 구 단위 정렬 정확도:")
    print(f"  Precision: {100*precision:.2f}%")
    print(f"  Recall: {100*recall:.2f}%")
    print(f"  F1 Score: {100*f1:.2f}% ⬅️ 핵심 지표")
    print(f"  평균 유사도: {100*np.mean(all_sims):.2f}%")
    print(f"  High Sim (≥0.9): {100*high_sim_count/total_pred_segs:.1f}%")
    
    print("\n" + "="*60)
    
    # 결과 저장
    results = {
        'matched_sentences': len(matched_pairs),
        'coverage': len(matched_pairs) / len(gold_map),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'avg_similarity': float(np.mean(all_sims)),
        'high_sim_rate': high_sim_count / total_pred_segs if total_pred_segs > 0 else 0,
        'seg_count_match_rate': seg_count_matches / len(matched_pairs) if matched_pairs else 0
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {output_file}")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S2P 평가 (원문 텍스트 매칭)')
    parser.add_argument('gold', help='Gold 파일')
    parser.add_argument('pred', help='Prediction 파일')
    parser.add_argument('-o', '--output', default='test_results/s2p_eval_textmatch.json')
    
    args = parser.parse_args()
    main(args.gold, args.pred, args.output)
