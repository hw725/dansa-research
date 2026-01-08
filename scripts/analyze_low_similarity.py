#!/usr/bin/env python3
"""avg_similarity < 0.6 케이스 분석"""

import json

def analyze_low_similarity(trace_path):
    """avg_similarity < 0.6인 후보 비율 확인"""
    
    total_candidates = 0
    low_sim_candidates = 0
    low_sim_selected = 0
    total_paragraphs = 0
    
    with open(trace_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj['stage'] != 'src_matched_selected':
                continue
            
            total_paragraphs += 1
            meta = obj.get('meta', {})
            top_cands = meta.get('top_candidates', [])
            best_tag = meta.get('best_tag', '')
            
            for cand in top_cands:
                if not cand.get('considered', False):
                    continue
                
                total_candidates += 1
                sim = cand.get('avg_similarity', 1.0)
                
                if sim < 0.6:
                    low_sim_candidates += 1
                    if cand['tag'] == best_tag:
                        low_sim_selected += 1
    
    print(f"=== avg_similarity < 0.6 분석 ===")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Total candidates: {total_candidates}")
    print(f"Low similarity (<0.6) candidates: {low_sim_candidates} ({low_sim_candidates/total_candidates*100:.1f}%)")
    print(f"Low similarity selected: {low_sim_selected}")
    print(f"\n→ Style bonus 무시 규칙이 적용된 비율: {low_sim_candidates/total_candidates*100:.1f}%")

if __name__ == '__main__':
    print("Alignment-guided 버전:")
    analyze_low_similarity('logs/pa_stage_trace_bthr0.72_thr0.72_ml10_seed1_alignment_guided.jsonl')
