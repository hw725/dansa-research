#!/usr/bin/env python3
"""SA 경계 기반 F1 평가 - 원문 정확 일치 subset에서 번역문 F1 측정

PA 방식: 원문(src) 경계 정확 일치 케이스만 필터 → 번역문(tgt) 경계 F1 계산
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import difflib
import pandas as pd
from typing import List, Tuple, Set
import warnings
warnings.filterwarnings("ignore")

from sa.io_manager import process_file


def _norm(s: str) -> str:
    """공백 제거 정규화"""
    return re.sub(r'[\s\u3000]', '', str(s))


def _boundary_positions(segments: List[str]) -> Set[int]:
    """세그먼트 리스트에서 경계 위치(누적 정규화 길이) 집합 반환"""
    positions = set()
    cursor = 0
    for i, seg in enumerate(segments):
        cursor += len(_norm(seg))
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions


def _prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, Recall, F1"""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _similarity(a: str, b: str) -> float:
    a_n, b_n = _norm(a), _norm(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def evaluate_sa_boundaries(pred_df, gold_df, sent_ids: List) -> dict:
    """원문 정확 일치 subset에서 번역문 경계 F1 측정"""
    
    # 1. 원문 경계 정확 일치 케이스 필터링
    src_exact_ids = []
    for sent_id in sent_ids:
        gold = gold_df[gold_df['문장식별자'] == sent_id]
        pred = pred_df[pred_df['문장식별자'] == sent_id]
        if gold.empty or pred.empty:
            continue
        gold_bounds = _boundary_positions([str(r) for r in gold['원문']])
        pred_bounds = _boundary_positions([str(r) for r in pred['원문']])
        if gold_bounds == pred_bounds:
            src_exact_ids.append(sent_id)
    
    print(f"📍 원문 정확 일치: {len(src_exact_ids)}/{len(sent_ids)} ({len(src_exact_ids)/len(sent_ids):.1%})")
    
    # 2. 원문 일치 subset에서 번역문 경계 F1 및 세그먼트별 유사도
    tp = fp = fn = 0
    seg_sims = []  # 각 세그먼트 쌍의 유사도
    
    for sent_id in src_exact_ids:
        gold = gold_df[gold_df['문장식별자'] == sent_id]
        pred = pred_df[pred_df['문장식별자'] == sent_id]
        
        gold_tgt = [str(r) for r in gold['번역문']]
        pred_tgt = [str(r) for r in pred['번역문']]
        
        gold_bounds = _boundary_positions(gold_tgt)
        pred_bounds = _boundary_positions(pred_tgt)
        
        tp += len(gold_bounds & pred_bounds)
        fp += len(pred_bounds - gold_bounds)
        fn += len(gold_bounds - pred_bounds)
        
        # 세그먼트별 유사도 (1:1 매칭되는 경우만)
        if len(gold_tgt) == len(pred_tgt):
            for g, p in zip(gold_tgt, pred_tgt):
                seg_sims.append(_similarity(g, p))
    
    precision, recall, f1 = _prf1(tp, fp, fn)
    
    return {
        'src_exact_count': len(src_exact_ids),
        'src_exact_rate': len(src_exact_ids) / len(sent_ids) if sent_ids else 0,
        'tgt_boundary_f1': f1,
        'tgt_boundary_precision': precision,
        'tgt_boundary_recall': recall,
        'avg_seg_similarity': sum(seg_sims) / len(seg_sims) if seg_sims else 0,
        'seg_sim_count': len(seg_sims),
        'total': len(sent_ids),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--run-sa", action="store_true")
    args = parser.parse_args()
    
    input_path = Path("datasets/pa/test.csv")
    gold_path = Path("datasets/sa/test.csv")
    output_path = Path("test_results/sa_eval_output.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🎯 SA 번역문 경계 F1 평가 (원문 일치 subset)")
    print("=" * 60)
    
    input_df = pd.read_csv(input_path)
    gold_df = pd.read_csv(gold_path)
    
    sent_ids = list(input_df['문장식별자'].unique()[:args.sample_size])
    sample_input = input_df[input_df['문장식별자'].isin(sent_ids)].copy()
    print(f"📂 샘플: {len(sent_ids)}문장")
    
    if args.run_sa:
        print("\n🚀 SA 실행 중...")
        temp_input = output_path.parent / "eval_input.xlsx"
        sample_input.to_excel(temp_input, index=False)
        
        success = process_file(
            input_file=str(temp_input),
            output_file=str(output_path),
            embedder_name='bge',
            max_workers=4,
            chunk_size=50,
            verbose=False,
            use_boundary_model=True,  # 🆕 경계 모델 사용
            dp_window=3,
            boundary_bonus=0.2,
            particle_bonus=0.3,
            length_penalty=0.08,
            sim_gamma=1.0,
        )
        if not success:
            print("❌ SA 실패")
            return 1
        pred_df = pd.read_excel(output_path)
    else:
        if not output_path.exists():
            print(f"❌ {output_path} 없음, --run-sa 사용")
            return 1
        pred_df = pd.read_excel(output_path)
    
    print(f"✅ Pred: {len(pred_df)}개 구")
    
    m = evaluate_sa_boundaries(pred_df, gold_df, sent_ids)
    
    print("\n" + "=" * 60)
    print("📈 결과 (원문 일치 subset → 번역문 F1)")
    print("=" * 60)
    print(f"  원문 정확 일치:   {m['src_exact_rate']:.1%} ({m['src_exact_count']}/{m['total']})")
    print(f"  번역문 경계 F1:   {m['tgt_boundary_f1']:.4f}")
    print(f"  번역문 Precision: {m['tgt_boundary_precision']:.4f}")
    print(f"  번역문 Recall:    {m['tgt_boundary_recall']:.4f}")
    print(f"  번역문 유사도:    {m['avg_tgt_similarity']:.4f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
