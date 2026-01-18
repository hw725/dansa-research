#!/usr/bin/env python3
"""클린 데이터만 사용하여 Cross-Attention 경계 모델 평가"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import difflib
import pandas as pd
from typing import List, Set
import warnings
warnings.filterwarnings("ignore")


def _norm(s: str) -> str:
    return re.sub(r'[\s\u3000]', '', str(s))


def _boundary_positions(segments: List[str]) -> Set[int]:
    positions = set()
    cursor = 0
    for i, seg in enumerate(segments):
        cursor += len(_norm(seg))
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions


def _prf1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main():
    # 싱글톤 리셋
    import common.sa_crossattn_boundary_loader as loader
    loader._crossattn_tagger_instance = None
    
    from common.sa_crossattn_boundary_loader import get_crossattn_boundary_tagger
    tagger = get_crossattn_boundary_tagger()
    
    gold_df = pd.read_csv("datasets/sa/test.csv")
    sent_ids = list(gold_df['문장식별자'].unique()[:100])
    
    print("=" * 60)
    print("🎯 클린 데이터 Cross-Attention 경계 모델 평가")
    print("=" * 60)
    
    total_phrases = 0
    clean_phrases = 0
    tp = fp = fn = 0
    seg_similarities = []
    
    for sent_id in sent_ids:
        gold_rows = gold_df[gold_df['문장식별자'] == sent_id].sort_values('구식별자')
        
        if gold_rows.empty:
            continue
        
        gold_src_segs = [str(r).strip() for r in gold_rows['원문']]
        gold_tgt_segs = [str(r).strip() for r in gold_rows['번역문']]
        
        total_phrases += len(gold_src_segs)
        
        # 원문 구에 공백이 없는 경우만 처리
        all_clean = all(' ' not in src for src in gold_src_segs if src)
        
        if not all_clean:
            continue
        
        clean_phrases += len(gold_src_segs)
        
        src_text = ' '.join(gold_src_segs)
        tgt_text = ' '.join(gold_tgt_segs)
        n_segs = len(gold_tgt_segs)
        
        pred_tgt_segs = tagger.segment_text(src_text, tgt_text, n_segments=n_segs)
        
        if len(pred_tgt_segs) == len(gold_tgt_segs):
            for gold_seg, pred_seg in zip(gold_tgt_segs, pred_tgt_segs):
                gold_norm = _norm(gold_seg)
                pred_norm = _norm(pred_seg)
                if gold_norm == pred_norm:
                    seg_similarities.append(1.0)
                elif gold_norm and pred_norm:
                    sim = difflib.SequenceMatcher(None, gold_norm, pred_norm).ratio()
                    seg_similarities.append(sim)
                else:
                    seg_similarities.append(0.0)
        
        gold_bounds = _boundary_positions(gold_tgt_segs)
        pred_bounds = _boundary_positions(pred_tgt_segs)
        
        tp += len(gold_bounds & pred_bounds)
        fp += len(pred_bounds - gold_bounds)
        fn += len(gold_bounds - pred_bounds)
    
    precision, recall, f1 = _prf1(tp, fp, fn)
    avg_sim = sum(seg_similarities) / len(seg_similarities) if seg_similarities else 0
    exact_match = sum(1 for s in seg_similarities if s == 1.0)
    
    print(f"\n총 구 수: {total_phrases}")
    print(f"클린 구 수: {clean_phrases} ({clean_phrases/total_phrases*100:.1f}%)")
    print(f"\n번역문 경계 F1:    {f1:.4f}")
    print(f"번역문 Precision:  {precision:.4f}")
    print(f"번역문 Recall:     {recall:.4f}")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    if seg_similarities:
        print(f"\n📊 세그먼트 유사도:")
        print(f"   평균 유사도:    {avg_sim:.4f}")
        print(f"   완전 일치:      {exact_match}/{len(seg_similarities)} ({exact_match/len(seg_similarities)*100:.1f}%)")


if __name__ == "__main__":
    main()
