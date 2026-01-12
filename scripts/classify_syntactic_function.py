#!/usr/bin/env python3
"""
Syntactic Function 자동 분류

Strategy 1 + 3 병행:
1. 기존 언어학 연구 매핑 적용 (configs/syntactic_function_mapping.json)
2. kiwipiepy 규칙 기반 분류
3. 통계 생성

사용법:
    python scripts/classify_syntactic_function.py \
        --csv hyeonto/reports/recluster_k16_child/reclustered.csv \
        --mapping configs/syntactic_function_mapping.json \
        --out-csv hyeonto/datasets/reclustered_with_syntax.csv
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

try:
    from kiwipiepy import Kiwi
    KIWI_AVAILABLE = True
except ImportError:
    KIWI_AVAILABLE = False
    print("⚠️ kiwipiepy not available - using mapping only")


def extract_hyeonto_markers(text: str) -> str:
    """src 텍스트에서 한글 현토 마커 추출

    Args:
        text: src_left 또는 src_right 텍스트

    Returns:
        추출된 한글 마커 (빈 문자열 가능)
    """
    if pd.isna(text):
        return ''
    # 한글만 추출 (한자 뒤에 붙은 토씨)
    matches = re.findall(r'[\u3131-\u318E\uAC00-\uD7A3]+', str(text))
    return ''.join(matches) if matches else ''


def classify_with_mapping(marker: str, mapping: Dict[str, str]) -> Optional[str]:
    """매핑 테이블로 분류

    Args:
        marker: 현토 마커 (예: "니", "라", "되")
        mapping: syntactic_function_mapping.json의 mappings

    Returns:
        분류된 syntactic_function 또는 None (매핑 실패)
    """
    return mapping.get(marker, None)


def classify_with_rules(marker: str, tgt: str, kiwi: Optional['Kiwi'] = None) -> str:
    """규칙 기반 분류 (kiwipiepy 활용)

    Args:
        marker: 현토 마커
        tgt: 번역 텍스트 (문맥 파악용)
        kiwi: Kiwi 인스턴스 (선택)

    Returns:
        분류된 syntactic_function
    """
    # NaN 처리
    if pd.isna(tgt):
        tgt = ''
    tgt = str(tgt)

    # 의문 종결
    if marker in ['니', '가', 'ㄴ가', '뇨', 'ㄹ까', 'ㄴ저'] or '?' in tgt:
        return '의문종결'

    # 평서 종결
    if marker in ['라', '니라', '도다', '다', 'ㄴ다']:
        return '평서종결'

    # 시제 (과거/미래)
    if marker in ['러', '러라', '러니', '더', '더니', '더라']:
        return '과거시제'
    if marker in ['리', '리라', '리오']:
        return '미래시제'

    # 피동
    if marker in ['되', '이']:
        # kiwipiepy로 수동태 문맥 확인
        if kiwi and KIWI_AVAILABLE:
            try:
                analyzed = kiwi.analyze(tgt, top_n=1)
                if analyzed:
                    morphemes = analyzed[0][0]
                    # XSV (피동 파생 접미사) 태그 확인
                    if any(m[1] == 'XSV' for m in morphemes):
                        return '피동'
            except:
                pass
        # 기본값
        if marker == '되':
            return '피동'
        elif marker == '이':
            return '주격조사'

    # 접속/연결
    if marker in ['고', '며', '하며']:
        return '접속연결'

    # 조건/양보
    if marker in ['면', '거든', '거니와', 'ㄹ진대']:
        return '조건연결'
    if marker in ['거늘']:
        return '대조연결'

    # 인용
    if marker in ['댄']:
        return '인용연결'

    # 원인
    if marker in ['지라', '디니']:
        return '원인연결'

    # 조사
    if marker in ['는']:
        return '주제조사'
    if marker in ['를']:
        return '목적격조사'
    if marker in ['의']:
        return '관형격조사'
    if marker in ['에']:
        return '처격조사'
    if marker in ['으로']:
        return '도구격조사'
    if marker in ['와']:
        return '접속조사'
    if marker in ['야']:
        return '호격조사'
    if marker in ['도']:
        return '보조사'
    if marker in ['라도']:
        return '양보조사'
    if marker in ['니라도']:
        return '조건조사'

    # 높임
    if marker in ['시', '샤']:
        return '주체높임'

    # 명사형
    if marker in ['ㅁ', '기', 'ㄴ디']:
        return '명사형어미'

    # 관형화
    if marker in ['디', 'ㄹ셰', 'ㄹ뎐']:
        return '관형화어미'

    # 시간 연결
    if marker in ['매']:
        return '시간연결'

    # 기간 표시
    if marker in ['ㄴ지']:
        return '기간표시'

    # 기타 (분류 불가)
    return '기타'


def main() -> int:
    p = argparse.ArgumentParser(description="Syntactic Function 자동 분류")
    p.add_argument("--csv", type=Path, required=True, help="입력 CSV 파일")
    p.add_argument("--mapping", type=Path, required=True, help="Mapping JSON 파일")
    p.add_argument("--out-csv", type=Path, required=True, help="출력 CSV 파일")
    p.add_argument("--mode", choices=['mapping', 'auto_classify', 'both'],
                   default='both', help="분류 모드 (기본: both)")
    p.add_argument("--use-kiwi", action='store_true',
                   help="kiwipiepy 사용 (피동 등 문맥 분류)")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1

    if not args.mapping.exists():
        print(f"❌ 파일 없음: {args.mapping}")
        return 1

    # 출력 디렉토리 생성
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 매핑 테이블 로드
    print(f"📄 매핑 테이블 로드: {args.mapping}")
    with open(args.mapping, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    mappings = mapping_data.get('mappings', {})

    print(f"✅ 매핑 테이블: {len(mappings)}개 마커")

    # kiwipiepy 초기화
    kiwi = None
    if args.use_kiwi and KIWI_AVAILABLE:
        print("🔧 kiwipiepy 초기화 중...")
        kiwi = Kiwi()
        print("✅ kiwipiepy 준비 완료")
    elif args.use_kiwi and not KIWI_AVAILABLE:
        print("⚠️ kiwipiepy 사용 불가 - 매핑만 사용")

    # CSV 로드
    print(f"📄 CSV 로드: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"✅ {len(df):,}행 로드")

    # src_left에서 마커 추출
    print("🔍 마커 추출 중...")
    df['marker'] = df['src_left'].apply(extract_hyeonto_markers)

    # 빈 마커 제거
    before_count = len(df)
    df = df[df['marker'] != '']
    after_count = len(df)
    print(f"✅ 마커 추출 완료: {after_count:,}행 ({before_count - after_count:,}행 제거)")
    print(f"   고유 마커: {df['marker'].nunique():,}개")

    # syntactic_function 분류
    print(f"🔬 Syntactic Function 분류 중 (모드: {args.mode})...")

    syntactic_functions = []
    mapping_count = 0
    rule_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="분류"):
        marker = row['marker']
        tgt = row.get('tgt_left', '')

        syntactic_func = None

        # Step 1: 매핑 테이블 적용
        if args.mode in ['mapping', 'both']:
            syntactic_func = classify_with_mapping(marker, mappings)
            if syntactic_func:
                mapping_count += 1

        # Step 2: 규칙 기반 분류 (매핑 실패 시)
        if syntactic_func is None and args.mode in ['auto_classify', 'both']:
            syntactic_func = classify_with_rules(marker, tgt, kiwi)
            rule_count += 1

        # Step 3: 기본값
        if syntactic_func is None:
            syntactic_func = '기타'

        syntactic_functions.append(syntactic_func)

    df['syntactic_function'] = syntactic_functions

    # 통계
    print("\n📊 분류 통계:")
    print(f"   매핑 테이블 적용: {mapping_count:,}개 ({mapping_count / len(df) * 100:.1f}%)")
    print(f"   규칙 기반 분류: {rule_count:,}개 ({rule_count / len(df) * 100:.1f}%)")
    print(f"   기타: {(syntactic_functions.count('기타')):,}개")
    print(f"\n   고유 syntactic_function: {df['syntactic_function'].nunique():,}개")

    # Top 10 syntactic_function
    print("\n🔝 Top 10 Syntactic Functions:")
    top_funcs = df['syntactic_function'].value_counts().head(10)
    for func, count in top_funcs.items():
        print(f"   {func}: {count:,}개 ({count / len(df) * 100:.1f}%)")

    # CSV 저장
    print(f"\n💾 CSV 저장: {args.out_csv}")
    df.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 저장 완료: {len(df):,}행")

    # 통계 파일 생성
    stats_path = args.out_csv.parent / f"{args.out_csv.stem}_stats.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("# Syntactic Function 분류 통계\n\n")
        f.write(f"입력 파일: {args.csv}\n")
        f.write(f"출력 파일: {args.out_csv}\n")
        f.write(f"분류 모드: {args.mode}\n")
        f.write(f"kiwipiepy 사용: {args.use_kiwi and KIWI_AVAILABLE}\n\n")
        f.write(f"총 행 수: {len(df):,}\n")
        f.write(f"고유 마커: {df['marker'].nunique():,}개\n")
        f.write(f"고유 syntactic_function: {df['syntactic_function'].nunique():,}개\n\n")
        f.write(f"매핑 테이블 적용: {mapping_count:,}개 ({mapping_count / len(df) * 100:.1f}%)\n")
        f.write(f"규칙 기반 분류: {rule_count:,}개 ({rule_count / len(df) * 100:.1f}%)\n")
        f.write(f"기타: {syntactic_functions.count('기타'):,}개\n\n")
        f.write("## Top 20 Syntactic Functions:\n\n")
        for func, count in df['syntactic_function'].value_counts().head(20).items():
            f.write(f"{func}: {count:,}개 ({count / len(df) * 100:.1f}%)\n")

    print(f"📄 통계 파일 저장: {stats_path}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
