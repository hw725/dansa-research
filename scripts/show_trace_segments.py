#!/usr/bin/env python3
"""trace JSONL에서 특정 문단의 src/tgt segments 확인"""

import json

def show_segments(trace_path, pid=10, stage='src_matched_selected'):
    with open(trace_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj['paragraph_id'] == pid and obj['stage'] == stage:
                print(f"=== {stage} stage, paragraph {pid} ===")
                print(f"\nBook: {obj.get('book_name', 'N/A')}")
                
                src_segs = obj.get('src_segments', [])
                tgt_segs = obj.get('tgt_segments', [])
                
                print(f"\nSource segments ({len(src_segs)}):")
                for i, seg in enumerate(src_segs, 1):
                    text = seg if isinstance(seg, str) else seg.get('text', seg.get('norm', ''))
                    print(f"  {i:2d}. {text[:60]}")
                
                print(f"\nTarget segments ({len(tgt_segs)}):")
                for i, seg in enumerate(tgt_segs, 1):
                    text = seg if isinstance(seg, str) else seg.get('text', seg.get('norm', ''))
                    print(f"  {i:2d}. {text[:60]}")
                
                # meta 정보
                meta = obj.get('meta', {})
                if 'best_tag' in meta:
                    print(f"\nBest tag: {meta['best_tag']}, score: {meta.get('best_score', 0):.4f}")
                
                break

if __name__ == '__main__':
    import sys
    trace_path = sys.argv[1] if len(sys.argv) > 1 else 'logs/pa_stage_trace_bthr0.72_thr0.72_ml10_seed1_fixed.jsonl'
    show_segments(trace_path)
