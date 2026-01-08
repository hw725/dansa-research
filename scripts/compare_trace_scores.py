#!/usr/bin/env python3
"""기존 vs fixed 버전의 trace 후보 점수 비교"""

import json
import sys

def compare_traces(old_path, new_path, pid=10):
    """첫 문단의 src_matched_selected stage 후보 점수 비교"""
    
    # 기존 버전
    with open(old_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj['stage'] == 'src_matched_selected' and obj['paragraph_id'] == pid:
                print('=== 기존 버전 (candidates trace) ===')
                print(f"Best tag: {obj['meta']['best_tag']}, score: {obj['meta']['best_score']:.4f}")
                print('\nTop 5 candidates:')
                for cand in obj['meta']['top_candidates'][:5]:
                    tag = cand['tag']
                    score = cand['score']
                    sim = cand.get('avg_similarity', 0)
                    prior = cand.get('prior_bonus', 0)
                    style = cand.get('boundary_style_bonus', 0)
                    print(f"  {tag:25s}: score={score:7.4f}, sim={sim:.4f}, prior={prior:+.4f}, style={style:+.4f}")
                break
    
    # Fixed 버전
    with open(new_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj['stage'] == 'src_matched_selected' and obj['paragraph_id'] == pid:
                print('\n=== Fixed 버전 (empty penalty 수정 + style disabled) ===')
                print(f"Best tag: {obj['meta']['best_tag']}, score: {obj['meta']['best_score']:.4f}")
                print('\nTop 5 candidates:')
                for cand in obj['meta']['top_candidates'][:5]:
                    tag = cand['tag']
                    score = cand['score']
                    sim = cand['avg_similarity']
                    prior = cand.get('prior_bonus', 0)
                    style = cand.get('boundary_style_bonus', 0)
                    # whitespace_dp penalty 상세
                    ws_severe = cand.get('penalty_ws_severe', 0)
                    ws_very_short = cand.get('penalty_ws_very_short', 0)
                    ws_ratio = cand.get('penalty_ws_ratio_outlier', 0)
                    ws_longest = cand.get('penalty_ws_longest_shortest', 0)
                    total_ws_penalty = ws_severe + ws_very_short + ws_ratio + ws_longest
                    
                    print(f"  {tag:25s}: score={score:7.4f}, sim={sim:.4f}, prior={prior:+.4f}, style={style:+.4f}, ws_penalty={total_ws_penalty:.4f}")
                break

if __name__ == '__main__':
    old = 'logs/pa_stage_trace_bthr0.72_thr0.72_ml10_seed1_candidates.jsonl'
    new = 'logs/pa_stage_trace_bthr0.72_thr0.72_ml10_seed1_fixed.jsonl'
    compare_traces(old, new)
