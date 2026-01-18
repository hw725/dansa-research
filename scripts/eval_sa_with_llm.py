#!/usr/bin/env python3
"""SA 파이프라인 평가: Cross-Attention + LLM 보정"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# LLM 보정 활성화
os.environ["USE_LLM_BOUNDARY_VERIFY"] = "1"
os.environ["LLM_BOUNDARY_BACKEND"] = "gemini"  # gemini 사용

import re
import pandas as pd
from typing import List, Set
import warnings
warnings.filterwarnings("ignore")

from sa.sa_aligner import process_single_row
from sa.io_manager import safe_process_sa_row
from common.sa_crossattn_boundary_loader import get_crossattn_boundary_tagger

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
    print("🔧 모델 로드 중...")
    if not hasattr(safe_process_sa_row, '_boundary_model'):
        safe_process_sa_row._boundary_model = get_crossattn_boundary_tagger()
    print("✅ 모델 로드 완료")
    print(f"📡 LLM 백엔드: {os.getenv('LLM_BOUNDARY_BACKEND', 'ollama')}")

    gold_df = pd.read_csv("datasets/sa/test.csv")
    sent_ids = list(gold_df['문장식별자'].unique()[:100])
    
    clean_sents = []
    for sent_id in sent_ids:
        gold_rows = gold_df[gold_df['문장식별자'] == sent_id].sort_values('구식별자')
        if gold_rows.empty: continue
        
        gold_src_segs = [str(r).strip() for r in gold_rows['원문']]
        gold_tgt_segs = [str(r).strip() for r in gold_rows['번역문']]
        
        if all(' ' not in src for src in gold_src_segs if src):
             clean_sents.append({
                'sent_id': sent_id,
                'src_text': ' '.join(gold_src_segs),
                'tgt_text': ' '.join(gold_tgt_segs),
                'tgt_segs': gold_tgt_segs,
            })
    
    # 속도를 위해 10개만 샘플링
    clean_sents = clean_sents[:10]
    print(f"평가 대상: {len(clean_sents)}개 문장")
    print("=" * 60)

    tp = fp = fn = 0
    llm_applied = 0
    
    for i, sent in enumerate(clean_sents):
        row_data = {
            '문장식별자': sent['sent_id'],
            '원문': sent['src_text'],
            '번역문': sent['tgt_text'],
        }
        
        try:
            result_rows = process_single_row(
                row_data, 
                use_boundary_model=True,
                boundary_threshold=0.5,
            )
            pred_tgt_segs = [r['번역문'] for r in result_rows]
            method = result_rows[0].get('분할방법', '') if result_rows else ''
            
            if '+llm_refine' in method:
                llm_applied += 1
            
            print(f"[{i+1}/{len(clean_sents)}] ID={sent['sent_id']}: {len(pred_tgt_segs)}개 세그먼트, 방법: {method}")
            
        except Exception as e:
            print(f"Error {sent['sent_id']}: {e}")
            pred_tgt_segs = [sent['tgt_text']]

        gold_bounds = _boundary_positions(sent['tgt_segs'])
        pred_bounds = _boundary_positions(pred_tgt_segs)
        
        tp += len(gold_bounds & pred_bounds)
        fp += len(pred_bounds - gold_bounds)
        fn += len(gold_bounds - pred_bounds)

    p, r, f1 = _prf1(tp, fp, fn)
    print("=" * 60)
    print(f"\n📊 결과 ({len(clean_sents)}개 문장):")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nLLM 보정 적용: {llm_applied}/{len(clean_sents)}개 문장")

if __name__ == "__main__":
    main()
