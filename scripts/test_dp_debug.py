"""DP 디버그 테스트"""
import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/pa')

import pandas as pd
from processor import process_paragraph_alignment_with_boundary_model
from common.boundary_model_loader import BoundaryModelLoader
from common.boundary_aware_alignment_loader import BoundaryAwareAlignmentMatcher

boundary_model = BoundaryModelLoader('/workspace/models/boundary_multitask.pt')
alignment_model = BoundaryAwareAlignmentMatcher('/workspace/models/dual_encoder_boundary_aware_pa.pt')

df = pd.read_excel('/workspace/test_results/pa_test_input_30.xlsx')
first_row = df.iloc[0]
src = first_row['원문']
tgt = first_row['번역문']

print(f'src len: {len(src)}, tgt len: {len(tgt)}')

result = process_paragraph_alignment_with_boundary_model(
    src_paragraph=src,
    tgt_paragraph=tgt,
    boundary_model=boundary_model,
    alignment_model=alignment_model,
    threshold=0.72,
    boundary_bonus_factor=2.0,
    shift_penalty_factor=0.0002,
)
print(f'result count: {len(result)}')
