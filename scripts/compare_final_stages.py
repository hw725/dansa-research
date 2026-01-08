#!/usr/bin/env python3
"""기존 vs fixed 버전의 final stage 비교"""

import json

def compare_final_stages(old_path, new_path, pid=10):
    """final stage 비교"""
    
    # Fixed 버전
    with open(new_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj['paragraph_id'] == pid and obj['stage'] == 'final':
                src_segs_new = obj['src_segments']
                tgt_segs_new = obj['tgt_segments']
                print(f"=== Fixed final (pid={pid}) ===")
                print(f"Source segments: {len(src_segs_new)}")
                print(f"Target segments: {len(tgt_segs_new)}")
                break
    
    # 기존 버전
    with open(old_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj['paragraph_id'] == pid and obj['stage'] == 'final':
                src_segs_old = obj['src_segments']
                tgt_segs_old = obj['tgt_segments']
                print(f"\n=== 기존 final (pid={pid}) ===")
                print(f"Source segments: {len(src_segs_old)}")
                print(f"Target segments: {len(tgt_segs_old)}")
                break
    
    # 차이 비교
    if src_segs_new != src_segs_old:
        print("\n⚠️ Source segments가 다릅니다!")
        print("\n기존 버전 source:")
        for i, seg in enumerate(src_segs_old, 1):
            text = seg if isinstance(seg, str) else seg.get('text', '')
            print(f"  {i:2d}. {text[:70]}")
        print("\nFixed 버전 source:")
        for i, seg in enumerate(src_segs_new, 1):
            text = seg if isinstance(seg, str) else seg.get('text', '')
            print(f"  {i:2d}. {text[:70]}")
    else:
        print("\n✅ Source segments가 동일합니다.")

if __name__ == '__main__':
    old = 'logs/pa_stage_trace_bthr0.72_thr0.72_ml10_seed1_candidates.jsonl'
    new = 'logs/pa_stage_trace_bthr0.72_thr0.72_ml10_seed1_fixed.jsonl'
    compare_final_stages(old, new)
