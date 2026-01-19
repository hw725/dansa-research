#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""현토 패턴별 체계적 가설 검증 (v6)

각 현토 패턴에 대해:
1. 기본 통계 (조건부 확률)
2. 영가설 테스트 (랜덤 섞기)
3. 반대가설 테스트 (다른 현토 비교)

사용 예:
    python scripts/validate_hyeonto_patterns_v6.py \
        --csv hyeonto/reports/sentence_boundary_v6_full/boundary_clusters.csv \
        --out-dir hyeonto/reports/hyeonto_pattern_validation_v6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# 검증할 현토 패턴과 대응 번역문 패턴
PATTERN_HYPOTHESES = {
    '리오': {
        'name': '-리오 (반어/설의)',
        'hypothesis': '-리오는 반어/설의 의문("~겠는가")과 대응한다',
        'translation_patterns': ['겠는가', '겠습니까', '있겠는가', '수있겠'],
        'category': '서법',
    },
    '러니': {
        'name': '-러니 (회상)',
        'hypothesis': '-러니는 회상/경험 서술("~더니", "~었더니")과 대응한다',
        'translation_patterns': ['더니', '었더니', '았더니', '했더니'],
        'category': '상',
    },
    '니라': {
        'name': '-니라 (서술/단정)',
        'hypothesis': '-니라는 단정 서술("~것이다", "~이다")과 대응한다',
        'translation_patterns': ['것이다', '이다.', '니다.', '있다.'],
        'category': '서법',
    },
    '잇가': {
        'name': '-잇가 (의문)',
        'hypothesis': '-잇가는 의문문("~는가", "~습니까")과 대응한다',
        'translation_patterns': ['는가', '습니까', '니까', '있는가'],
        'category': '서법',
    },
    '하니': {
        'name': '-하니 (이유/연결)',
        'hypothesis': '-하니는 이유/연결("~하니", "~하므로", "~때문에")과 대응한다',
        'translation_patterns': ['하니', '하므로', '때문에', '하였으니'],
        'category': '연결',
    },
    '되': {
        'name': '-되 (인용/전환)',
        'hypothesis': '-되는 인용/전환("라고", "다만", "그러나")과 대응한다',
        'translation_patterns': ['라고', '다만', '그러나', '하되'],
        'category': '표점',
    },
}


def has_pattern_in_src(row, pattern):
    """src에서 패턴 존재 확인"""
    src = str(row.get('src_left', '')) + str(row.get('src_right', ''))
    return pattern in src


def has_pattern_in_tgt(row, patterns):
    """tgt에서 패턴 존재 확인"""
    tgt = str(row.get('tgt_right', ''))
    return any(p in tgt for p in patterns)


def validate_pattern(df, hyeonto_pattern, trans_patterns, n_iterations=100, seed=42):
    """단일 패턴에 대한 체계적 검증"""
    
    # 조건 생성
    has_hyeonto = df.apply(lambda r: has_pattern_in_src(r, hyeonto_pattern), axis=1)
    has_trans = df.apply(lambda r: has_pattern_in_tgt(r, trans_patterns), axis=1)
    
    # 2x2 분할표
    a = (has_hyeonto & has_trans).sum()
    b = (has_hyeonto & ~has_trans).sum()
    c = (~has_hyeonto & has_trans).sum()
    d = (~has_hyeonto & ~has_trans).sum()
    
    n_hyeonto = a + b
    n_total = len(df)
    
    if n_hyeonto == 0:
        return None
    
    # 조건부 확률
    p_trans_given_hyeonto = a / n_hyeonto * 100
    p_trans_given_not_hyeonto = c / (c + d) * 100 if (c + d) > 0 else 0
    
    # Chi-square
    try:
        chi2, p_value, dof, expected = chi2_contingency([[a, b], [c, d]])
        cramers_v = np.sqrt(chi2 / n_total)
    except:
        chi2, p_value, cramers_v = 0, 1, 0
    
    # 영가설 테스트: 랜덤 섞기
    np.random.seed(seed)
    random_rates = []
    for _ in range(n_iterations):
        shuffled = np.random.permutation(has_hyeonto.values)
        a_rand = (shuffled & has_trans.values).sum()
        n_rand = shuffled.sum()
        rate = a_rand / n_rand * 100 if n_rand > 0 else 0
        random_rates.append(rate)
    
    mean_rand = np.mean(random_rates)
    std_rand = np.std(random_rates)
    z_score = (p_trans_given_hyeonto - mean_rand) / std_rand if std_rand > 0 else 0
    null_rejected = z_score > 3
    
    # 반대가설: 다른 현토들과 비교
    other_patterns = [p for p in PATTERN_HYPOTHESES.keys() if p != hyeonto_pattern]
    alt_rates = {}
    for other in other_patterns[:3]:  # 상위 3개만
        other_mask = df.apply(lambda r: has_pattern_in_src(r, other), axis=1)
        if other_mask.sum() > 0:
            rate = (other_mask & has_trans).sum() / other_mask.sum() * 100
            alt_rates[other] = rate
    
    max_alt_rate = max(alt_rates.values()) if alt_rates else 0
    alt_rejected = p_trans_given_hyeonto > max_alt_rate * 1.5  # 50% 이상 높아야 기각
    
    return {
        'hyeonto': hyeonto_pattern,
        'n_cases': int(n_hyeonto),
        'contingency': {'a': int(a), 'b': int(b), 'c': int(c), 'd': int(d)},
        'p_trans_given_hyeonto': float(p_trans_given_hyeonto),
        'p_trans_given_not_hyeonto': float(p_trans_given_not_hyeonto),
        'difference': float(p_trans_given_hyeonto - p_trans_given_not_hyeonto),
        'chi2': float(chi2),
        'p_value': float(p_value),
        'cramers_v': float(cramers_v),
        'null_test': {
            'mean_random': float(mean_rand),
            'std_random': float(std_rand),
            'z_score': float(z_score),
            'rejected': bool(null_rejected),
        },
        'alt_test': {
            'other_rates': {k: float(v) for k, v in alt_rates.items()},
            'max_alt_rate': float(max_alt_rate),
            'rejected': bool(alt_rejected),
        },
        'hypothesis_supported': bool(null_rejected and alt_rejected),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="현토 패턴별 체계적 가설 검증")
    p.add_argument("--csv", type=Path, default=Path("hyeonto/reports/sentence_boundary_v6_full/boundary_clusters.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/hyeonto_pattern_validation_v6"))
    p.add_argument("--iterations", type=int, default=100)
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 CSV 로드: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"📊 데이터: {len(df):,}행")

    print(f"\n{'='*70}")
    print(f" 현토 패턴별 체계적 가설 검증")
    print(f"{'='*70}")

    results = {}
    
    for pattern, info in PATTERN_HYPOTHESES.items():
        print(f"\n### {info['name']} ###")
        print(f"가설: {info['hypothesis']}")
        
        result = validate_pattern(df, pattern, info['translation_patterns'], args.iterations)
        
        if result:
            results[pattern] = result
            results[pattern]['info'] = info
            
            print(f"  - 건수: {result['n_cases']:,}")
            print(f"  - P(번역패턴|현토) = {result['p_trans_given_hyeonto']:.1f}%")
            print(f"  - P(번역패턴|~현토) = {result['p_trans_given_not_hyeonto']:.1f}%")
            print(f"  - χ² = {result['chi2']:.1f}, V = {result['cramers_v']:.3f}")
            
            null_status = "✅ 기각" if result['null_test']['rejected'] else "❌ 기각실패"
            alt_status = "✅ 기각" if result['alt_test']['rejected'] else "❌ 기각실패"
            hyp_status = "✅ 지지" if result['hypothesis_supported'] else "❌ 기각"
            
            print(f"  - 영가설: {null_status} (z={result['null_test']['z_score']:.1f})")
            print(f"  - 반대가설: {alt_status}")
            print(f"  - 최종: {hyp_status}")
        else:
            print(f"  ⚠️ 데이터 부족")

    # 결과 저장
    print(f"\n{'='*70}")
    print(f" 결과 저장")
    print(f"{'='*70}")

    # JSON
    summary_json = args.out_dir / "pattern_validation_results.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {summary_json}")

    # 마크다운 보고서
    report_md = args.out_dir / "PATTERN_VALIDATION_REPORT.md"
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write("# 현토 패턴별 체계적 가설 검증 보고서 (v6)\n\n")
        f.write(f"**분석일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**데이터**: {args.csv.name} ({len(df):,}건)\n\n")
        
        f.write("---\n\n")
        f.write("## 검증 방법론\n\n")
        f.write("각 현토 패턴에 대해 다음 3단계 검증을 수행:\n\n")
        f.write("1. **기본 통계**: 조건부 확률 P(번역패턴|현토), Chi-square 검정\n")
        f.write("2. **영가설 테스트**: 현토를 랜덤 섞어도 동일한 상관관계가 나타나는가?\n")
        f.write("3. **반대가설 테스트**: 다른 현토도 동일한 번역패턴과 연관되는가?\n\n")
        
        f.write("---\n\n")
        f.write("## 검증 결과 요약\n\n")
        f.write("| 현토 | 가설 | P(번역\\|현토) | z-score | 영가설 | 반대가설 | 최종 |\n")
        f.write("|------|------|:---:|:---:|:---:|:---:|:---:|\n")
        
        for pattern, result in results.items():
            info = result['info']
            null = "✅" if result['null_test']['rejected'] else "❌"
            alt = "✅" if result['alt_test']['rejected'] else "❌"
            final = "✅" if result['hypothesis_supported'] else "❌"
            f.write(f"| {info['name']} | {info['hypothesis'][:30]}... | {result['p_trans_given_hyeonto']:.1f}% | {result['null_test']['z_score']:.1f} | {null} | {alt} | {final} |\n")
        
        f.write("\n---\n\n")
        
        # 각 패턴별 상세
        for pattern, result in results.items():
            info = result['info']
            f.write(f"## {info['name']}\n\n")
            f.write(f"**가설**: {info['hypothesis']}\n\n")
            f.write(f"**범주**: {info['category']}\n\n")
            
            f.write("### 1. 기본 통계\n\n")
            f.write("| 조건 | 번역패턴 O | 번역패턴 X |\n")
            f.write("|------|:---:|:---:|\n")
            c = result['contingency']
            f.write(f"| 현토 O | {c['a']:,} | {c['b']:,} |\n")
            f.write(f"| 현토 X | {c['c']:,} | {c['d']:,} |\n\n")
            
            f.write(f"- **P(번역패턴 \\| 현토)** = {result['p_trans_given_hyeonto']:.1f}%\n")
            f.write(f"- **P(번역패턴 \\| ~현토)** = {result['p_trans_given_not_hyeonto']:.1f}%\n")
            f.write(f"- **차이** = {result['difference']:.1f}%p\n")
            f.write(f"- **χ²** = {result['chi2']:.1f}, **Cramér's V** = {result['cramers_v']:.3f}\n\n")
            
            f.write("### 2. 영가설 테스트 (랜덤 섞기)\n\n")
            nt = result['null_test']
            f.write(f"- 원본 P(번역\\|현토) = {result['p_trans_given_hyeonto']:.1f}%\n")
            f.write(f"- 랜덤 평균 = {nt['mean_random']:.1f}%, 표준편차 = {nt['std_random']:.2f}%\n")
            f.write(f"- **z-score = {nt['z_score']:.1f}**\n")
            status = "✅ **영가설 기각** (우연이 아님)" if nt['rejected'] else "❌ 영가설 기각 실패"
            f.write(f"- 결과: {status}\n\n")
            
            f.write("### 3. 반대가설 테스트 (다른 현토 비교)\n\n")
            at = result['alt_test']
            f.write("| 다른 현토 | P(번역패턴) |\n")
            f.write("|----------|:---:|\n")
            for other, rate in at['other_rates'].items():
                f.write(f"| -{other} | {rate:.1f}% |\n")
            f.write(f"\n- 본 현토: {result['p_trans_given_hyeonto']:.1f}%, 최대 다른 현토: {at['max_alt_rate']:.1f}%\n")
            status = "✅ **반대가설 기각** (본 현토가 독보적)" if at['rejected'] else "❌ 반대가설 기각 실패"
            f.write(f"- 결과: {status}\n\n")
            
            f.write("### 4. 최종 판정\n\n")
            if result['hypothesis_supported']:
                f.write(f"✅ **가설 지지**: \"{info['hypothesis']}\"\n\n")
            else:
                f.write(f"❌ **가설 기각 또는 불충분**\n\n")
            
            f.write("---\n\n")
        
        f.write("## 결론\n\n")
        supported = sum(1 for r in results.values() if r['hypothesis_supported'])
        total = len(results)
        f.write(f"**{total}개 패턴 중 {supported}개에서 가설이 통계적으로 지지됨**\n\n")
        
        for pattern, result in results.items():
            info = result['info']
            status = "✅" if result['hypothesis_supported'] else "❌"
            f.write(f"- {status} {info['name']}: {info['hypothesis']}\n")

    print(f"  ✅ {report_md}")

    print(f"\n✅ 체계적 가설 검증 완료!")
    print(f"   - 검증 패턴: {len(results)}개")
    print(f"   - 가설 지지: {supported}/{total}")
    
    return 0


if __name__ == "__main__":
    exit(main())
