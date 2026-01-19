import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/pa')

import pandas as pd
from pathlib import Path
import json
import logging
import os

# 1. 설정 및 모델 로드
config = json.load(open('/workspace/csp_config.json'))
pa_params = config.get('pa_selection_params', {})

from common.boundary_model_loader import BoundaryModelLoader
from common.boundary_aware_alignment_loader import BoundaryAwareAlignmentMatcher
from p2s.processor import process_paragraph_alignment_with_boundary_model

boundary_model = BoundaryModelLoader(
    model_path=Path('/workspace/models/boundary_multitask.pt'),
    device='cuda'
)
alignment_matcher = BoundaryAwareAlignmentMatcher(
    model_path=Path('/workspace/models/dual_encoder_boundary_aware_pa.pt'),
    device='cuda',
    boundary_weight=0.1
)

# 2. 문단 56 데이터 준비
input_df = pd.read_excel('/workspace/test_results/ex250_sample_100_seed5/pa_test_input_seed5.xlsx')
para_56 = input_df[input_df['문단식별자'] == 56].iloc[0]
src_text = str(para_56['원문'])
tgt_text = str(para_56['번역문'])

print(f"\n--- 문단 56 후보 부족 테스트 (th=0.3) ---")

trace_path = Path("/workspace/test_results/para56_trace_th0.3.jsonl")

# th=0.3으로 낮춰서 후보를 많이 뽑습니다.
results = process_paragraph_alignment_with_boundary_model(
    src_paragraph=src_text,
    tgt_paragraph=tgt_text,
    boundary_model=boundary_model,
    alignment_model=alignment_matcher,
    threshold=0.3, # 0.72 -> 0.3
    verbose=True,
    dp_debug_out=str(trace_path)
)

print("\n=== 결과 확인 (th=0.3) ===")
# GOLD 비교 위치 확인
# GOLD S4: ... 避雷霆之威하며 不畏權臣之禍하니 (141)
# GOLD S5: 也라 能忘其身而愛陛下者也어늘 (151)
for i, s in enumerate(results):
    print(f"S{i+1}: {s}")
