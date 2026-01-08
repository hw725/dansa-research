#!/usr/bin/env python3
"""
Boundary 정보를 추가한 Context-aware Alignment 데이터 생성

입력: datasets/alignment/pa/train.jsonl (144,686개)
출력: datasets/alignment/pa/train_boundary_aware.jsonl

추가 필드:
- src_boundaries: List[int] - 원문 어절 경계 위치 (character index)
- tgt_boundaries: List[int] - 번역문 어절/구절 경계 위치
- boundary_match: int - 경계 일치 여부 (1: 일치, 0: 불일치)

목표:
- positive + 경계 일치: label=1, boundary_match=1
- positive + 경계 불일치: label=1, boundary_match=0 (hard negative)
- negative: label=0, boundary_match=0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"


def extract_boundaries(text: str, is_src: bool = True) -> List[int]:
    """
    텍스트에서 어절/구절 경계 위치 추출
    
    Args:
        text: 입력 텍스트
        is_src: True면 한문 원문 (공백 기준), False면 한국어 번역문 (공백+구두점 기준)
    
    Returns:
        경계 character index 리스트 (0-indexed, 시작 위치 포함)
    """
    if not text:
        return []
    
    boundaries = [0]  # 시작 위치
    
    if is_src:
        # 원문: 공백으로 구분된 어절 경계
        pos = 0
        for match in re.finditer(r'\s+', text):
            boundary_pos = match.end()
            if boundary_pos < len(text):
                boundaries.append(boundary_pos)
            pos = boundary_pos
    else:
        # 번역문: 공백 + 주요 구두점 (쉼표, 마침표, 물음표 등)
        pos = 0
        # 구두점 뒤 또는 공백 뒤를 경계로 간주
        for match in re.finditer(r'[\s,\.!?\)\]\}]+', text):
            boundary_pos = match.end()
            if boundary_pos < len(text):
                boundaries.append(boundary_pos)
            pos = boundary_pos
    
    return sorted(set(boundaries))


def create_hard_negatives_from_positive(
    src: str, 
    tgt: str, 
    src_boundaries: List[int],
    tgt_boundaries: List[int],
    next_src: str = None
) -> List[Dict]:
    """
    positive pair로부터 boundary 불일치 hard negative 생성
    
    전략:
    1. 경계 쉬프트: src를 다음 문장 일부와 합쳐 경계를 틀리게 만듦
    2. 부분 매칭: src의 일부만 사용하여 경계 불일치 생성
    """
    hard_negs = []
    
    # 전략 1: 경계 쉬프트 (next_src 있을 때)
    if next_src and next_src.strip():
        # next_src의 첫 어절만 가져오기
        next_words = next_src.split()
        if next_words:
            shifted_src = src + " " + next_words[0]
            shifted_boundaries = extract_boundaries(shifted_src, is_src=True)
            
            hard_negs.append({
                "src": shifted_src,
                "tgt": tgt,
                "src_boundaries": shifted_boundaries,
                "tgt_boundaries": tgt_boundaries,
                "label": 1,  # 의미는 유사
                "boundary_match": 0  # 경계는 불일치
            })
    
    # 전략 2: 부분 매칭 (src의 80% 또는 120% 사용)
    src_words = src.split()
    if len(src_words) > 2:
        # 80% 케이스: 마지막 어절 제거
        partial_src = " ".join(src_words[:-1])
        partial_boundaries = extract_boundaries(partial_src, is_src=True)
        
        hard_negs.append({
            "src": partial_src,
            "tgt": tgt,
            "src_boundaries": partial_boundaries,
            "tgt_boundaries": tgt_boundaries,
            "label": 1,
            "boundary_match": 0
        })
    
    return hard_negs


def main() -> int:
    parser = argparse.ArgumentParser(description="Boundary-aware alignment 데이터 생성")
    parser.add_argument(
        "--input",
        default=str(DATASETS_ROOT / "alignment" / "pa" / "train.jsonl"),
        help="입력 train.jsonl 경로"
    )
    parser.add_argument(
        "--output",
        default=str(DATASETS_ROOT / "alignment" / "pa" / "train_boundary_aware.jsonl"),
        help="출력 경로"
    )
    parser.add_argument(
        "--add-hard-neg",
        action="store_true",
        help="Hard negative 샘플 추가 (경계 불일치)"
    )
    parser.add_argument(
        "--hard-neg-ratio",
        type=float,
        default=0.3,
        help="Hard negative 비율 (0~1, 기본 0.3)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일 없음: {input_path}")
    
    # 안전장치: test/val 파일을 입력으로 넣는 사고 방지
    lower = str(input_path).replace("\\", "/").lower()
    if "/test" in lower or "/val" in lower or lower.endswith("test.jsonl") or lower.endswith("val.jsonl"):
        raise ValueError(
            f"❌ 학습 데이터 생성에는 train 파일만 사용할 수 있습니다.\n"
            f"입력 파일: {input_path}\n"
            f"train.jsonl 파일을 지정하세요."
        )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📖 입력: {input_path}")
    print(f"📝 출력: {output_path}")
    print(f"⚙️  Hard negative 추가: {args.add_hard_neg}")
    print(f"⚙️  Hard negative 비율: {args.hard_neg_ratio}")
    print()
    
    # 데이터 로드
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"✅ 로드 완료: {len(samples):,}개 샘플")
    
    # Boundary 정보 추가
    output_samples = []
    
    for i, sample in enumerate(tqdm(samples, desc="Processing")):
        src = sample.get("src", "")
        tgt = sample.get("tgt", "")
        label = sample.get("label", 1)
        
        if not src or not tgt:
            continue
        
        # Boundary 추출
        src_boundaries = extract_boundaries(src, is_src=True)
        tgt_boundaries = extract_boundaries(tgt, is_src=False)
        
        # 기본 positive 샘플 (경계 일치)
        output_sample = {
            "book": sample.get("book", ""),
            "src": src,
            "tgt": tgt,
            "src_boundaries": src_boundaries,
            "tgt_boundaries": tgt_boundaries,
            "label": label,
            "boundary_match": 1 if label == 1 else 0
        }
        output_samples.append(output_sample)
        
        # Hard negative 생성 (일부 샘플만)
        if args.add_hard_neg and label == 1 and i < len(samples) - 1:
            import random
            if random.random() < args.hard_neg_ratio:
                next_sample = samples[i + 1]
                next_src = next_sample.get("src", "")
                
                # 같은 책인지 확인
                if (sample.get("book") == next_sample.get("book") and 
                    next_src.strip()):
                    
                    hard_negs = create_hard_negatives_from_positive(
                        src, tgt, src_boundaries, tgt_boundaries, next_src
                    )
                    output_samples.extend(hard_negs)
    
    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in output_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print()
    print(f"✅ 저장 완료: {len(output_samples):,}개 샘플")
    print(f"   - 원본: {len(samples):,}개")
    print(f"   - 추가: {len(output_samples) - len(samples):,}개 (hard negatives)")
    print(f"📁 {output_path}")
    
    # 통계
    label_1 = sum(1 for s in output_samples if s.get("label") == 1)
    boundary_1 = sum(1 for s in output_samples if s.get("boundary_match") == 1)
    
    print()
    print("📊 통계:")
    print(f"   - label=1 (positive): {label_1:,}개 ({label_1/len(output_samples)*100:.1f}%)")
    print(f"   - boundary_match=1: {boundary_1:,}개 ({boundary_1/len(output_samples)*100:.1f}%)")
    print(f"   - hard negatives (label=1, boundary=0): {label_1 - boundary_1:,}개")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
