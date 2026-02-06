"""
현토 마커 정규화 누락 분석

전체 코퍼스에서 현토 마커를 추출하고,
정규화가 필요하지만 아직 정의되지 않은 변이형 쌍을 찾습니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import regex
from typing import List, Dict, Tuple

from hyeonto_normalizer import (
    normalize_hyeonto_marker, 
    NORMALIZATION_TABLE,
    JOSA_NORMALIZATION,
    EOMI_NORMALIZATION,
    COMPLEX_NORMALIZATION
)


def extract_all_markers(df: pd.DataFrame, cols: List[str] = None) -> Counter:
    """데이터프레임에서 모든 현토 마커 추출"""
    
    if cols is None:
        # 문자열 컬럼 자동 선택
        cols = df.select_dtypes(include=['object']).columns.tolist()
    
    marker_counter = Counter()
    
    for col in cols:
        if col not in df.columns:
            continue
        
        for text in df[col].dropna():
            # 한글만 추출 (유니코드 프로퍼티 사용)
            markers = regex.findall(r'\p{Hangul}+', str(text))
            marker_counter.update(markers)
    
    return marker_counter


def find_potential_variants(markers: Counter, min_count: int = 10) -> List[Dict]:
    """
    잠재적 변이형 쌍 탐지
    
    규칙:
    1. 음운 조건 변이 (을/를, 은/는, 으X/X)
    2. 모음 조화 변이 (아/어, 았/었)
    3. 선행 음절 변이 (이X/X, 인X/은X/ㄴX)
    """
    
    marker_list = [m for m, c in markers.items() if c >= min_count]
    
    potential_pairs = []
    
    # 1. 을/를 패턴
    for m in marker_list:
        if '을' in m:
            variant = m.replace('을', '를')
            if variant in marker_list and variant != m:
                if m not in NORMALIZATION_TABLE:
                    potential_pairs.append({
                        'type': '을/를',
                        'variant1': m,
                        'variant2': variant,
                        'count1': markers[m],
                        'count2': markers[variant],
                        'suggested_repr': variant  # 를을 대표형으로
                    })
    
    # 2. 은/는 패턴
    for m in marker_list:
        if '은' in m and m not in ['은', '은혜']:  # 단독 '은'은 제외
            variant = m.replace('은', '는')
            if variant in marker_list and variant != m:
                if m not in NORMALIZATION_TABLE:
                    potential_pairs.append({
                        'type': '은/는',
                        'variant1': m,
                        'variant2': variant,
                        'count1': markers[m],
                        'count2': markers[variant],
                        'suggested_repr': variant
                    })
    
    # 3. 으X/X 패턴 (으로/로, 으니/니, 으면/면 등)
    for m in marker_list:
        if m.startswith('으') and len(m) > 1:
            variant = m[1:]  # '으' 제거
            if variant in marker_list:
                if m not in NORMALIZATION_TABLE:
                    potential_pairs.append({
                        'type': '으X/X',
                        'variant1': m,
                        'variant2': variant,
                        'count1': markers[m],
                        'count2': markers[variant],
                        'suggested_repr': variant
                    })
    
    # 4. 이X/X 패턴 (이라/라, 이니/니 등)
    for m in marker_list:
        if m.startswith('이') and len(m) > 1:
            variant = m[1:]
            if variant in marker_list and len(variant) >= 1:
                if m not in NORMALIZATION_TABLE:
                    potential_pairs.append({
                        'type': '이X/X',
                        'variant1': m,
                        'variant2': variant,
                        'count1': markers[m],
                        'count2': markers[variant],
                        'suggested_repr': variant
                    })
    
    # 5. 인X/은X/ㄴX 패턴
    for m in marker_list:
        if m.startswith('인') and len(m) > 1:
            variant_eun = '은' + m[1:]
            variant_nieun = 'ㄴ' + m[1:]
            
            if variant_eun in marker_list:
                if m not in NORMALIZATION_TABLE:
                    potential_pairs.append({
                        'type': '인X/은X',
                        'variant1': m,
                        'variant2': variant_eun,
                        'count1': markers[m],
                        'count2': markers[variant_eun],
                        'suggested_repr': variant_nieun if variant_nieun in marker_list else m
                    })
            
            if variant_nieun in marker_list:
                if m not in NORMALIZATION_TABLE:
                    potential_pairs.append({
                        'type': '인X/ㄴX',
                        'variant1': m,
                        'variant2': variant_nieun,
                        'count1': markers[m],
                        'count2': markers[variant_nieun],
                        'suggested_repr': variant_nieun
                    })
    
    # 6. 하야/하여 패턴
    for m in marker_list:
        if '야' in m and '하' in m:
            variant = m.replace('야', '여')
            if variant in marker_list and variant != m:
                if m not in NORMALIZATION_TABLE:
                    potential_pairs.append({
                        'type': '야/여',
                        'variant1': m,
                        'variant2': variant,
                        'count1': markers[m],
                        'count2': markers[variant],
                        'suggested_repr': variant
                    })
    
    # 7. 았/었 패턴
    for m in marker_list:
        if '았' in m:
            variant = m.replace('았', '었')
            if variant in marker_list and variant != m:
                if m not in NORMALIZATION_TABLE:
                    potential_pairs.append({
                        'type': '았/었',
                        'variant1': m,
                        'variant2': variant,
                        'count1': markers[m],
                        'count2': markers[variant],
                        'suggested_repr': variant  # '었'을 대표형으로
                    })
    
    # 8. 과/와 패턴
    for m in marker_list:
        if '과' in m:
            variant = m.replace('과', '와')
            if variant in marker_list and variant != m:
                if m not in NORMALIZATION_TABLE:
                    potential_pairs.append({
                        'type': '과/와',
                        'variant1': m,
                        'variant2': variant,
                        'count1': markers[m],
                        'count2': markers[variant],
                        'suggested_repr': variant
                    })
    
    # 중복 제거
    seen = set()
    unique_pairs = []
    for pair in potential_pairs:
        key = tuple(sorted([pair['variant1'], pair['variant2']]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
    
    return unique_pairs


def analyze_uncovered_markers(markers: Counter, min_count: int = 50) -> List[Dict]:
    """
    정규화 테이블에 없지만 빈도가 높은 마커 분석
    """
    
    uncovered = []
    
    for marker, count in markers.most_common():
        if count < min_count:
            break
        
        # 이미 정규화 테이블에 있으면 건너뜀
        if marker in NORMALIZATION_TABLE:
            continue
        
        # 정규화 후 변화가 없는 마커도 확인
        normalized = normalize_hyeonto_marker(marker)
        if normalized == marker:
            uncovered.append({
                'marker': marker,
                'count': count,
                'normalized': normalized,
                'needs_review': True
            })
    
    return uncovered


def main():
    base_dir = Path(__file__).parent
    reports_dir = base_dir / "reports"
    
    print("=" * 70)
    print("? 현토 마커 정규화 누락 분석")
    print("=" * 70)
    
    # 데이터 로드
    data_paths = [
        reports_dir / "phrase_k4_normalized" / "phrase_clusters.csv",
        reports_dir / "sentence_k4_normalized" / "sentence_clusters.csv"
    ]
    
    all_markers = Counter()
    
    for path in data_paths:
        if path.exists():
            print(f"\n? Loading: {path.name}")
            df = pd.read_csv(path)
            
            # 마커 추출 (원문, left_sentence, right_sentence 등)
            text_cols = [c for c in df.columns if '원문' in c or 'sentence' in c.lower() or 'src' in c.lower()]
            markers = extract_all_markers(df, text_cols)
            all_markers.update(markers)
            print(f"   → {len(markers):,}개 고유 마커 추출")
    
    print(f"\n? 총 고유 마커: {len(all_markers):,}개")
    print(f"? 총 마커 빈도: {sum(all_markers.values()):,}회")
    
    # 현재 정규화 테이블 통계
    print(f"\n? 현재 정규화 테이블:")
    print(f"   ? 조사 정규화: {len(JOSA_NORMALIZATION)}개")
    print(f"   ? 어미 정규화: {len(EOMI_NORMALIZATION)}개")
    print(f"   ? 복합형 정규화: {len(COMPLEX_NORMALIZATION)}개")
    print(f"   ? 총: {len(NORMALIZATION_TABLE)}개")
    
    # 잠재적 변이형 탐지
    print("\n" + "=" * 70)
    print("? 잠재적 변이형 쌍 탐지 (min_count=10)")
    print("=" * 70)
    
    potential_pairs = find_potential_variants(all_markers, min_count=10)
    
    if potential_pairs:
        # 타입별 그룹화
        by_type = {}
        for pair in potential_pairs:
            t = pair['type']
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(pair)
        
        for ptype, pairs in sorted(by_type.items()):
            print(f"\n▶ {ptype} 패턴 ({len(pairs)}개)")
            print("-" * 60)
            
            # 빈도 합계 순으로 정렬
            pairs_sorted = sorted(pairs, key=lambda x: x['count1'] + x['count2'], reverse=True)
            
            for pair in pairs_sorted[:15]:
                total = pair['count1'] + pair['count2']
                print(f"   '{pair['variant1']}' ({pair['count1']:,}) ↔ "
                      f"'{pair['variant2']}' ({pair['count2']:,}) "
                      f"→ 제안: '{pair['suggested_repr']}' [합계: {total:,}]")
    else:
        print("   추가 변이형 쌍 없음")
    
    # 고빈도 미정규화 마커 분석
    print("\n" + "=" * 70)
    print("? 고빈도 미정규화 마커 (min_count=100)")
    print("=" * 70)
    
    uncovered = analyze_uncovered_markers(all_markers, min_count=100)
    
    print(f"\n▶ 검토 필요 마커: {len(uncovered)}개")
    print("-" * 60)
    
    # 상위 50개만 출력
    for item in uncovered[:50]:
        print(f"   '{item['marker']}': {item['count']:,}회")
    
    # 결과 저장
    output_dir = reports_dir / "normalization_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 잠재적 변이형 저장
    if potential_pairs:
        pairs_df = pd.DataFrame(potential_pairs)
        pairs_df = pairs_df.sort_values('count1', ascending=False)
        pairs_path = output_dir / "potential_variant_pairs.csv"
        pairs_df.to_csv(pairs_path, index=False, encoding='utf-8-sig')
        print(f"\n? 잠재적 변이형 저장: {pairs_path}")
    
    # 미정규화 마커 저장
    uncovered_df = pd.DataFrame(uncovered)
    uncovered_path = output_dir / "uncovered_high_freq_markers.csv"
    uncovered_df.to_csv(uncovered_path, index=False, encoding='utf-8-sig')
    print(f"? 미정규화 마커 저장: {uncovered_path}")
    
    # 마크다운 리포트 생성
    report_lines = [
        "# 현토 마커 정규화 검토 리포트",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 1. 현행 정규화 테이블 현황",
        "",
        f"- 조사 정규화: {len(JOSA_NORMALIZATION)}개",
        f"- 어미 정규화: {len(EOMI_NORMALIZATION)}개",
        f"- 복합형 정규화: {len(COMPLEX_NORMALIZATION)}개",
        f"- **총**: {len(NORMALIZATION_TABLE)}개",
        "",
        "---",
        "",
        "## 2. 추가 정규화 필요 변이형",
        "",
    ]
    
    if potential_pairs:
        for ptype, pairs in sorted(by_type.items()):
            report_lines.append(f"### {ptype} 패턴 ({len(pairs)}개)")
            report_lines.append("")
            report_lines.append("| 변이형1 | 빈도1 | 변이형2 | 빈도2 | 제안 대표형 |")
            report_lines.append("|:--------|------:|:--------|------:|:-----------|")
            
            pairs_sorted = sorted(pairs, key=lambda x: x['count1'] + x['count2'], reverse=True)
            for pair in pairs_sorted[:20]:
                report_lines.append(
                    f"| `{pair['variant1']}` | {pair['count1']:,} | "
                    f"`{pair['variant2']}` | {pair['count2']:,} | "
                    f"`{pair['suggested_repr']}` |"
                )
            report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "## 3. 검토 필요 고빈도 마커",
        "",
        "| 마커 | 빈도 |",
        "|:-----|-----:|",
    ])
    
    for item in uncovered[:30]:
        report_lines.append(f"| `{item['marker']}` | {item['count']:,} |")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 4. 권장 조치",
        "",
        "1. 위 변이형 쌍 중 언어학적으로 동일한 기능을 하는 것들을 `hyeonto_normalizer.py`에 추가",
        "2. 고빈도 미정규화 마커 중 변이형이 있는지 추가 조사",
        "3. 정규화 적용 후 클러스터링 재수행 검토",
    ])
    
    report_path = output_dir / "NORMALIZATION_REVIEW_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding='utf-8')
    print(f"? 리포트 저장: {report_path}")
    
    print("\n" + "=" * 70)
    print("? 정규화 검토 분석 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
