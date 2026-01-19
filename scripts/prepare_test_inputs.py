#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테스트 데이터를 PA/SA 파이프라인 입력 형식(Excel)로 변환

평가 스크립트(evaluate_hierarchical_segmentation.py)에서 사용한 
테스트 데이터를 실제 파이프라인에서 실행 가능한 형식으로 변환합니다.

생성 파일:
- test_pd_input.xlsx: PD test 데이터 → PA 파이프라인 입력용
- test_pa_input.xlsx: PA test 데이터 → SA 파이프라인 입력용
"""

from pathlib import Path
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
OUTPUT_DIR = WORKSPACE_ROOT / "test_inputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("테스트 데이터 → 파이프라인 입력 형식 변환")
    print("=" * 60)
    
    # 1. PD test → PA 입력
    print("\n1️⃣  PD test → PA 입력 변환")
    pd_test_path = DATASETS_ROOT / "pd" / "test.csv"
    if not pd_test_path.exists():
        print(f"❌ {pd_test_path} 파일이 없습니다.")
        return
    
    pd_test = pd.read_csv(pd_test_path, dtype=str).fillna("")
    print(f"   로드: {len(pd_test)}개 문단")
    
    # PA 파이프라인은 '원문', '번역문' 컬럼 필요
    pd_input = pd_test[['src', 'tgt']].rename(columns={'src': '원문', 'tgt': '번역문'})
    pd_output_path = OUTPUT_DIR / "test_pd_input.xlsx"
    pd_input.to_excel(pd_output_path, index=False)
    print(f"   저장: {pd_output_path}")
    print(f"   사용: python p2s/main.py {pd_output_path} output_pa.xlsx --use-boundary-model")
    
    # 2. PA test → SA 입력
    print("\n2️⃣  PA test → SA 입력 변환")
    pa_test_path = DATASETS_ROOT / "pa" / "test.csv"
    if not pa_test_path.exists():
        print(f"❌ {pa_test_path} 파일이 없습니다.")
        return
    
    pa_test = pd.read_csv(pa_test_path, dtype=str).fillna("")
    print(f"   로드: {len(pa_test)}개 문장")
    
    # SA 파이프라인도 '원문', '번역문' 컬럼 필요
    pa_input = pa_test[['src', 'tgt']].rename(columns={'src': '원문', 'tgt': '번역문'})
    pa_output_path = OUTPUT_DIR / "test_pa_input.xlsx"
    pa_input.to_excel(pa_output_path, index=False)
    print(f"   저장: {pa_output_path}")
    print(f"   사용: python s2p/main.py {pa_output_path} output_sa.xlsx --use-boundary-model")
    
    # 3. 정답 데이터 참고
    print("\n3️⃣  정답 데이터 (비교용)")
    pa_test_path = DATASETS_ROOT / "pa" / "test.csv"
    sa_test_path = DATASETS_ROOT / "sa" / "test.csv"
    print(f"   PA 정답: {pa_test_path}")
    print(f"   SA 정답: {sa_test_path}")
    
    print("\n" + "=" * 60)
    print("✅ 변환 완료!")
    print("=" * 60)
    print("\n📝 다음 단계:")
    print("1. PA 파이프라인 실행:")
    print(f"   python p2s/main.py {pd_output_path} output_pa_baseline.xlsx")
    print(f"   python p2s/main.py {pd_output_path} output_pa_boundary.xlsx --use-boundary-model --boundary-threshold 0.4")
    print("\n2. SA 파이프라인 실행:")
    print(f"   python s2p/main.py {pa_output_path} output_sa_baseline.xlsx")
    print(f"   python s2p/main.py {pa_output_path} output_sa_boundary.xlsx --use-boundary-model --boundary-threshold 0.4")
    print("\n3. 결과 비교:")
    print(f"   - 기존 방식: output_pa_baseline.xlsx / output_sa_baseline.xlsx")
    print(f"   - 경계 모델: output_pa_boundary.xlsx / output_sa_boundary.xlsx")
    print(f"   - 정답 데이터: {pa_test_path} / {sa_test_path}")
    print("\n4. 평가 메트릭 확인:")
    print("   - 세그먼트 수 비교 (기존 vs 경계 모델 vs 정답)")
    print("   - text_similarity 계산")
    print("   - 수동 품질 검증 (샘플링)")


if __name__ == "__main__":
    main()
