#!/usr/bin/env python3
"""SA 무결성 및 기준선 성능 측정 스크립트

Usage:
    python scripts/verify_sa_baseline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sa.io_manager import process_file
from common.integrity_verifier import verify_global_integrity
from accuracy.sa_evaluator import AccuracyEvaluator


def main():
    # 경로 설정
    input_path = Path("datasets/pa/test.csv")
    gold_path = Path("datasets/sa/test.csv")
    output_path = Path("test_results/sa_baseline_output.xlsx")
    
    # 출력 디렉터리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 테스트 샘플 크기 (전체 테스트는 시간이 오래 걸림)
    SAMPLE_SIZE = 100
    
    print("=" * 80)
    print("🔍 SA 무결성 및 기준선 성능 측정")
    print("=" * 80)
    
    # 1. 입력 데이터 로드 (샘플)
    print(f"\n📂 입력 파일: {input_path}")
    input_df = pd.read_csv(input_path)
    print(f"   전체 행 수: {len(input_df):,}")
    
    # 샘플 추출 (문장식별자 기준 상위 N개)
    unique_sent_ids = input_df['문장식별자'].unique()[:SAMPLE_SIZE]
    sample_df = input_df[input_df['문장식별자'].isin(unique_sent_ids)].copy()
    print(f"   샘플 문장 수: {len(unique_sent_ids)}")
    print(f"   샘플 행 수: {len(sample_df)}")
    
    # 샘플 저장 (임시)
    sample_input_path = output_path.parent / "sa_sample_input.xlsx"
    sample_df.to_excel(sample_input_path, index=False)
    
    # 2. SA 실행
    print(f"\n🚀 SA 실행 중...")
    try:
        success = process_file(
            input_file=str(sample_input_path),
            output_file=str(output_path),
            embedder_name='bge',
            max_workers=4,
            chunk_size=50,
            use_parallel=True,
            verbose=False,
            # 기본 파라미터 사용
            dp_window=3,
            boundary_bonus=0.2,
            particle_bonus=0.3,
            length_penalty=0.08,
            sim_gamma=1.0,
        )
        
        if not success:
            print("❌ SA 실행 실패")
            return 1
            
    except Exception as e:
        print(f"❌ SA 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 3. 결과 로드
    result_df = pd.read_excel(output_path)
    print(f"✅ SA 완료: {len(result_df):,}개 구 생성")
    
    # 4. 무결성 검증
    print("\n" + "=" * 80)
    print("🔍 무결성 검증")
    print("=" * 80)
    
    passed, losses_df, analysis = verify_global_integrity(
        input_df=sample_df,
        result_df=result_df,
        source_col='원문',
        target_col='번역문',
        verbose=True
    )
    
    # 5. 정확도 평가 (Gold와 비교)
    print("\n" + "=" * 80)
    print("📊 정확도 평가 (Gold 대비)")
    print("=" * 80)
    
    # Gold 데이터에서 동일 문장 추출
    gold_df = pd.read_csv(gold_path)
    gold_sample = gold_df[gold_df['문장식별자'].isin(unique_sent_ids)].copy()
    
    print(f"   Gold 행 수: {len(gold_sample):,}")
    print(f"   Pred 행 수: {len(result_df):,}")
    
    # 문장별 구 수 비교
    gold_counts = gold_sample.groupby('문장식별자').size()
    pred_counts = result_df.groupby('문장식별자').size()
    
    exact_match = 0
    total = 0
    for sent_id in unique_sent_ids:
        gold_n = gold_counts.get(sent_id, 0)
        pred_n = pred_counts.get(sent_id, 0)
        if gold_n == pred_n:
            exact_match += 1
        total += 1
    
    segment_count_match_rate = exact_match / total if total > 0 else 0
    print(f"\n📈 세그먼트 수 일치율: {exact_match}/{total} ({segment_count_match_rate:.1%})")
    
    # 6. 결과 요약
    print("\n" + "=" * 80)
    print("📋 기준선 측정 완료")
    print("=" * 80)
    print(f"   무결성: {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"   원문 Δ: {analysis['source']['delta']:+}자")
    print(f"   번역문 Δ: {analysis['target']['delta']:+}자")
    print(f"   세그먼트 수 일치율: {segment_count_match_rate:.1%}")
    print(f"   출력 파일: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
