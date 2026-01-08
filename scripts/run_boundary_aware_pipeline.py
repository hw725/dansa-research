#!/usr/bin/env python3
"""
Boundary-aware Alignment Model 학습 및 평가 파이프라인

단계:
1. Boundary 정보 추가 데이터 생성
2. Context-aware model 학습
3. PA strict 평가
4. 기존 모델과 비교
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = WORKSPACE_ROOT / "scripts"
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
MODELS_ROOT = WORKSPACE_ROOT / "models"
TEST_RESULTS_ROOT = WORKSPACE_ROOT / "test_results"


def run_command(cmd: list, description: str):
    """명령어 실행 및 결과 출력"""
    print()
    print("=" * 80)
    print(f"🚀 {description}")
    print("=" * 80)
    print(f"$ {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=WORKSPACE_ROOT)
    
    if result.returncode != 0:
        print(f"❌ 실패: {description}")
        raise SystemExit(result.returncode)
    
    print(f"✅ 완료: {description}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Boundary-aware alignment 파이프라인")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="데이터 생성 단계 건너뛰기 (이미 생성된 경우)"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="학습 단계 건너뛰기 (이미 학습된 경우)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="학습 epochs (기본: 5)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="배치 크기 (기본: 64)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="최대 학습 스텝 (0=무제한, 기본: 0)"
    )
    parser.add_argument(
        "--add-hard-neg",
        action="store_true",
        help="Hard negative 샘플 추가"
    )
    parser.add_argument(
        "--hard-neg-ratio",
        type=float,
        default=0.3,
        help="Hard negative 비율 (기본: 0.3)"
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=0.3,
        help="Boundary loss 가중치 (기본: 0.3)"
    )
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 데이터 생성
    if not args.skip_data:
        data_cmd = [
            sys.executable,
            str(SCRIPTS_ROOT / "prepare_boundary_aware_data.py"),
            "--input",
            str(DATASETS_ROOT / "alignment" / "pa" / "train.jsonl"),
            "--output",
            str(DATASETS_ROOT / "alignment" / "pa" / "train_boundary_aware.jsonl"),
        ]
        
        if args.add_hard_neg:
            data_cmd.append("--add-hard-neg")
            data_cmd.extend(["--hard-neg-ratio", str(args.hard_neg_ratio)])
        
        run_command(data_cmd, "1/3: Boundary 정보 추가 데이터 생성")
    else:
        print("⏭️  데이터 생성 단계 건너뛰기")
    
    # 2. 모델 학습
    if not args.skip_train:
        train_cmd = [
            sys.executable,
            str(SCRIPTS_ROOT / "train_boundary_aware_alignment.py"),
            "--train-jsonl",
            str(DATASETS_ROOT / "alignment" / "pa" / "train_boundary_aware.jsonl"),
            "--epochs",
            str(args.epochs),
            "--batch",
            str(args.batch),
            "--max-steps",
            str(args.max_steps),
            "--boundary-weight",
            str(args.boundary_weight),
            "--output",
            str(MODELS_ROOT / "dual_encoder_boundary_aware_pa.pt"),
        ]
        
        run_command(train_cmd, "2/3: Context-aware Alignment Model 학습")
    else:
        print("⏭️  학습 단계 건너뛰기")
    
    # 3. 평가 (PA strict)
    # TODO: PA processor에서 boundary-aware model 사용하도록 수정 필요
    print()
    print("=" * 80)
    print("📊 3/3: 평가 및 비교")
    print("=" * 80)
    print()
    print("⚠️  다음 단계:")
    print("   1. pa/processor.py에서 BoundaryAwareAlignmentMatcher 사용")
    print("   2. evaluate_pa_accuracy.py로 평가 실행")
    print("   3. 기존 모델(F1=0.41)과 비교")
    print()
    print(f"📁 학습된 모델: {MODELS_ROOT / 'dual_encoder_boundary_aware_pa.pt'}")
    print(f"📁 데이터: {DATASETS_ROOT / 'alignment' / 'pa' / 'train_boundary_aware.jsonl'}")
    print()
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
