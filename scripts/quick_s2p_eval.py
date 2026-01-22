#!/usr/bin/env python3
"""
S2P 빠른 평가 스크립트
문장식별자별로 구 경계 정확도를 빠르게 측정
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
import argparse

def normalize_text(text):
    """텍스트 정규화 (공백/구두점 제거)"""
    if pd.isna(text):
        return ""
    import re
    s = str(text).strip()
    s = re.sub(r'[\s\t\n\r]+', '', s)
    return s

def calculate_similarity(text1, text2):
    """문자열 유사도 계산"""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()

def evaluate_s2p(gold_file, pred_file, sample_size=None):
    """S2P 평가 메인 함수"""
    # 파일 로드
    print(f"📂 Gold 파일 로딩: {gold_file}")
    if gold_file.endswith('.xlsx'):
        gold_df = pd.read_excel(gold_file)
    else:
        gold_df = pd.read_csv(gold_file)
    
    print(f"📂 Pred 파일 로딩: {pred_file}")
    if pred_file.endswith('.xlsx'):
        pred_df = pd.read_excel(pred_file)
    else:
        pred_df = pd.read_csv(pred_file)
    
    print(f"  Gold: {len(gold_df)} rows, Pred: {len(pred_df)} rows")
    
    # 문장식별자별 그룹화
    gold_grouped = defaultdict(list)
    for _, row in gold_df.iterrows():
        sent_id = int(row['문장식별자'])
        gold_grouped[sent_id].append({
            'src': normalize_text(row['원문']),
            'tgt': normalize_text(row['번역문'])
        })
    
    pred_grouped = defaultdict(list)
    for _, row in pred_df.iterrows():
        sent_id = int(row['문장식별자'])
        pred_grouped[sent_id].append({
            'src': normalize_text(row['원문']),
            'tgt': normalize_text(row['번역문'])
        })
    
    # 공통 문장식별자 찾기
    common_ids = set(gold_grouped.keys()) & set(pred_grouped.keys())
    if sample_size:
        common_ids = list(common_ids)[:sample_size]
    else:
        common_ids = list(common_ids)
    
    print(f"\n📊 평가 대상: {len(common_ids)} 문장")
    print(f"  Gold 고유 문장: {len(gold_grouped)}")
    print(f"  Pred 고유 문장: {len(pred_grouped)}")
    print(f"  공통 문장: {len(common_ids)}")
    
    # 평가 지표
    results = {
        'src_exact_match': [],      # 원문 정확히 일치 (문장 내 모든 구)
        'tgt_exact_match': [],      # 번역문 정확히 일치
        'src_similarity': [],       # 원문 전체 유사도
        'tgt_similarity': [],       # 번역문 전체 유사도
        'segment_count_match': [],  # 구 개수 일치
        'segment_count_diff': [],   # 구 개수 차이
    }
    
    mismatches = []
    
    for sent_id in common_ids:
        gold_segs = gold_grouped[sent_id]
        pred_segs = pred_grouped[sent_id]
        
        # 전체 텍스트 결합
        gold_src_full = ''.join([s['src'] for s in gold_segs])
        gold_tgt_full = ''.join([s['tgt'] for s in gold_segs])
        pred_src_full = ''.join([s['src'] for s in pred_segs])
        pred_tgt_full = ''.join([s['tgt'] for s in pred_segs])
        
        # 원문 무결성 (입력과 출력 원문이 같아야 함)
        src_exact = (gold_src_full == pred_src_full)
        results['src_exact_match'].append(1.0 if src_exact else 0.0)
        results['src_similarity'].append(calculate_similarity(gold_src_full, pred_src_full))
        
        # 번역문 비교
        tgt_exact = (gold_tgt_full == pred_tgt_full)
        results['tgt_exact_match'].append(1.0 if tgt_exact else 0.0)
        results['tgt_similarity'].append(calculate_similarity(gold_tgt_full, pred_tgt_full))
        
        # 구 개수 비교
        seg_match = (len(gold_segs) == len(pred_segs))
        results['segment_count_match'].append(1.0 if seg_match else 0.0)
        results['segment_count_diff'].append(len(pred_segs) - len(gold_segs))
        
        # 불일치 기록
        if not src_exact:
            mismatches.append({
                'sent_id': sent_id,
                'type': 'src_mismatch',
                'gold': gold_src_full[:100],
                'pred': pred_src_full[:100]
            })
    
    # 구 단위 F1 계산 (번역문 경계 기준)
    print("\n🔍 구 단위 F1 계산 중...")
    tp, fp, fn = 0, 0, 0
    high_sim_count = 0
    total_tgt_sims = []
    
    for sent_id in common_ids:
        gold_segs = gold_grouped[sent_id]
        pred_segs = pred_grouped[sent_id]
        
        gold_tgts = [s['tgt'] for s in gold_segs]
        pred_tgts = [s['tgt'] for s in pred_segs]
        
        # 간단한 매칭: 각 pred 세그먼트에 대해 가장 유사한 gold 찾기
        matched_gold = set()
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
            
            total_tgt_sims.append(best_sim)
            if best_sim >= 0.9:
                high_sim_count += 1
            
            if best_sim >= 0.8:  # 80% 이상 유사도면 매칭
                if best_idx >= 0:
                    matched_gold.add(best_idx)
                    tp += 1
            else:
                fp += 1
        
        fn += len(gold_tgts) - len(matched_gold)
    
    # F1 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 결과 출력
    print("\n" + "="*60)
    print("📊 S2P 평가 결과")
    print("="*60)
    
    print(f"\n📌 데이터 커버리지:")
    print(f"  평가된 문장 수: {len(common_ids)} / {len(gold_grouped)} ({100*len(common_ids)/len(gold_grouped):.1f}%)")
    
    print(f"\n📌 전역 텍스트 무결성:")
    print(f"  원문 Exact Match: {100*np.mean(results['src_exact_match']):.1f}%")
    print(f"  원문 평균 유사도: {100*np.mean(results['src_similarity']):.1f}%")
    print(f"  번역문 Exact Match: {100*np.mean(results['tgt_exact_match']):.1f}%")
    print(f"  번역문 평균 유사도: {100*np.mean(results['tgt_similarity']):.1f}%")
    
    print(f"\n📌 구 개수 정확도:")
    print(f"  구 개수 일치율: {100*np.mean(results['segment_count_match']):.1f}%")
    print(f"  구 개수 차이 (평균): {np.mean(results['segment_count_diff']):.2f}")
    
    print(f"\n📌 구 단위 정렬 정확도 (F1):")
    print(f"  Precision: {100*precision:.2f}%")
    print(f"  Recall: {100*recall:.2f}%")
    print(f"  F1 Score: {100*f1:.2f}% ⬅️ 핵심 지표")
    print(f"  평균 Tgt 유사도: {100*np.mean(total_tgt_sims):.2f}%")
    print(f"  High Sim (≥0.9): {100*high_sim_count/len(total_tgt_sims):.1f}%")
    
    print("\n" + "="*60)
    
    # 불일치 샘플 출력 (상위 5개만, 깔끔하게)
    if mismatches:
        print(f"\n⚠️ 원문 불일치: {len(mismatches)}건")
        for m in mismatches[:3]:
            gold_short = m['gold'][:40].replace('\n', ' ')
            pred_short = m['pred'][:40].replace('\n', ' ')
            print(f"  문장 {m['sent_id']}: G={gold_short}... P={pred_short}...")
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'src_exact_match': np.mean(results['src_exact_match']),
        'tgt_avg_sim': np.mean(total_tgt_sims),
        'high_sim_rate': high_sim_count / len(total_tgt_sims) if total_tgt_sims else 0
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S2P 빠른 평가')
    parser.add_argument('gold', help='Gold 파일 경로')
    parser.add_argument('pred', help='Prediction 파일 경로')
    parser.add_argument('--sample', type=int, help='샘플 문장 수 (빠른 테스트용)')
    parser.add_argument('--output', '-o', default='test_results/s2p_quick_eval.json', help='결과 저장 경로')
    
    args = parser.parse_args()
    results = evaluate_s2p(args.gold, args.pred, args.sample)
    
    # JSON으로 저장
    import json
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 결과 저장됨: {args.output}")
