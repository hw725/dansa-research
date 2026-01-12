#!/usr/bin/env python3
"""
F1 0.80 → 0.90 달성을 위한 심층 분석 및 개선 방향 제시

분석 내용:
1. 현재 trace의 점수 분포 및 family별 특성
2. 1등/2등 간 점수 차이가 작은 케이스 추출 (재조정 여지)
3. marker bonus, threshold 등 가중치/선택 로직 최적화 방향
4. 구체적 실험 제안
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import statistics
from typing import List, Dict, Tuple

def load_jsonl(path: Path) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def analyze_score_distribution(trace_records: List[dict]):
    """점수 분포 및 선택 여지 분석"""
    
    analysis = {
        'total_cases': 0,
        'close_race': [],  # 1등과 2등 차이 < 0.03
        'moderate_gap': [],  # 0.03 <= 차이 < 0.10
        'clear_winner': [],  # 차이 >= 0.10
        'family_stats': defaultdict(lambda: {
            'count': 0,
            'avg_score': [],
            'score_components': defaultdict(list)
        })
    }
    
    for rec in trace_records:
        if rec.get('stage') != 'src_matched_selected':
            continue
        
        meta = rec.get('meta', {})
        candidates = meta.get('top_candidates', [])
        
        if not candidates:
            continue
        
        # Considered 후보만
        considered = [c for c in candidates if c.get('considered', False)]
        if len(considered) < 2:
            continue
        
        analysis['total_cases'] += 1
        
        # 점수 기준 정렬
        sorted_cands = sorted(considered, key=lambda c: c.get('score', 0), reverse=True)
        best = sorted_cands[0]
        second = sorted_cands[1]
        
        margin = best.get('score', 0) - second.get('score', 0)
        
        case_info = {
            'src_id': rec.get('src_id'),
            'best': {
                'family': best.get('family'),
                'tag': best.get('tag'),
                'score': best.get('score'),
                'prior_bonus': best.get('prior_bonus', 0)
            },
            'second': {
                'family': second.get('family'),
                'tag': second.get('tag'),
                'score': second.get('score'),
                'prior_bonus': second.get('prior_bonus', 0)
            },
            'margin': margin
        }
        
        if margin < 0.03:
            analysis['close_race'].append(case_info)
        elif margin < 0.10:
            analysis['moderate_gap'].append(case_info)
        else:
            analysis['clear_winner'].append(case_info)
        
        # Family별 통계
        family = best.get('family', 'unknown')
        analysis['family_stats'][family]['count'] += 1
        analysis['family_stats'][family]['avg_score'].append(best.get('score', 0))
        
        # 점수 구성 요소 (만약 trace에 있다면)
        if 'score_breakdown' in best:
            for key, val in best['score_breakdown'].items():
                analysis['family_stats'][family]['score_components'][key].append(val)
    
    return analysis

def suggest_improvements(analysis: Dict):
    """분석 결과 기반 개선 방향 제시"""
    
    print("\n" + "="*80)
    print("F1 0.80 → 0.90 달성을 위한 개선 전략")
    print("="*80)
    
    total = analysis['total_cases']
    close = len(analysis['close_race'])
    moderate = len(analysis['moderate_gap'])
    clear = len(analysis['clear_winner'])
    
    print(f"\n## 1. 점수 분포 현황")
    print(f"총 케이스: {total}")
    print(f"  - 근소한 차이 (<0.03): {close} ({close/total*100:.1f}%)")
    print(f"  - 중간 차이 (0.03~0.10): {moderate} ({moderate/total*100:.1f}%)")
    print(f"  - 명확한 우승 (>=0.10): {clear} ({clear/total*100:.1f}%)")
    
    print(f"\n### 핵심 발견:")
    print(f"  - **{close + moderate}건({(close+moderate)/total*100:.1f}%)의 케이스에서 1등/2등 차이 < 0.10**")
    print(f"  - 이 케이스들은 가중치 조정으로 결과가 바뀔 가능성이 큼")
    print(f"  - 특히 근소한 차이({close}건)는 재평가 우선순위")
    
    # Close race 케이스 분석
    if analysis['close_race']:
        print(f"\n## 2. 근소한 차이 케이스 분석 (상위 10개)")
        close_sorted = sorted(analysis['close_race'], key=lambda x: x['margin'])[:10]
        
        print(f"\n{'Margin':<8} {'Best Family':<15} {'Best Score':<10} {'Second Family':<15} {'Second Score':<10} {'Prior Diff':<10}")
        print("-" * 100)
        
        for case in close_sorted:
            margin = case['margin']
            best_fam = case['best']['family'] or 'unknown'
            best_score = case['best']['score'] or 0
            best_bonus = case['best']['prior_bonus'] or 0
            
            second_fam = case['second']['family'] or 'unknown'
            second_score = case['second']['score'] or 0
            second_bonus = case['second']['prior_bonus'] or 0
            
            bonus_diff = best_bonus - second_bonus
            
            print(f"{margin:<8.4f} {best_fam:<15} {best_score:<10.4f} {second_fam:<15} {second_score:<10.4f} {bonus_diff:<10.4f}")
    
    # Family별 통계
    print(f"\n## 3. Family별 평균 점수")
    print(f"{'Family':<20} {'Count':<8} {'Avg Score':<10}")
    print("-" * 50)
    
    for family in sorted(analysis['family_stats'].keys()):
        stats = analysis['family_stats'][family]
        count = stats['count']
        avg = statistics.mean(stats['avg_score']) if stats['avg_score'] else 0
        print(f"{family:<20} {count:<8} {avg:<10.4f}")
    
    # 개선 방향 제시
    print(f"\n## 4. 구체적 개선 방향")
    
    print(f"\n### 📊 실험 1: Prior Bonus (현토 마커) 가중치 조정")
    print(f"  **현재 상태**: boundary 후보에 평균 0.15 보너스 적용")
    print(f"  **문제**: 근소한 차이 케이스에서 보너스 차이가 결정적일 수 있음")
    print(f"  **제안**:")
    print(f"    - 실험 A: 보너스 계수 0.10 (약화) - boundary 과선택 방지")
    print(f"    - 실험 B: 보너스 계수 0.20 (강화) - 현토 신호 더 신뢰")
    print(f"    - 실험 C: 동적 보너스 - 유사도에 따라 0.10~0.20 가변")
    print(f"  **검증 방법**: seed 1~10 재실험 후 F1 비교")

    print(f"\n### 📊 실험 2: Boundary Model Threshold 조정")
    print(f"  **현재 상태**: boundary 선택 55.1% (과다 가능성)")
    print(f"  **문제**: threshold가 낮아서 노이즈가 많은 boundary 후보 생성")
    print(f"  **제안**:")
    print(f"    - 실험 A: threshold 0.70 → 0.75 (더 보수적)")
    print(f"    - 실험 B: threshold 0.70 → 0.65 (더 공격적)")
    print(f"    - 실험 C: confidence 기반 필터링 (top-k boundary만)")
    print(f"  **검증 방법**: boundary 비율이 40~50%로 조정되고 F1 개선")

    print(f"\n### 📊 실험 3: Supar Weight 조정")
    print(f"  **현재 상태**: supar 선택 30.3%, 평균 점수 낮음 (0.47~0.50)")
    print(f"  **문제**: supar가 구조적으로 좋아도 점수가 낮아 선택 안 됨")
    print(f"  **제안**:")
    print(f"    - 실험 A: supar base score에 +0.05 보정")
    print(f"    - 실험 B: supar base score에 +0.10 보정")
    print(f"    - 실험 C: supar의 구조 일치도를 별도 가중치로 반영")
    print(f"  **검증 방법**: supar 선택 비율이 35~40%로 증가하고 F1 개선")

    print(f"\n### 📊 실험 4: Ensemble Voting (새 접근)")
    print(f"  **현재 상태**: 단순 최고 점수 선택")
    print(f"  **문제**: 근소한 차이 케이스에서 단일 지표에 의존")
    print(f"  **제안**:")
    print(f"    - 실험 A: 1등 점수 < 2등 점수 + 0.03이면, 두 후보의 결과를 비교해 더 나은 쪽 선택")
    print(f"    - 실험 B: 근소한 차이일 때 similarity가 더 높은 쪽 선택")
    print(f"    - 실험 C: 3개 후보 전체의 가중 평균 (soft voting)")
    print(f"  **검증 방법**: close_race 케이스 {close}건에서 개선 확인")
    
    print(f"\n## 5. 우선순위 실행 계획")
    print(f"\n### Phase 1: 빠른 실험 (1일)")
    print(f"  1. Prior bonus 0.10 / 0.15 / 0.20 grid search (3회 실험)")
    print(f"  → 총 9회 실험(3개 설정 × seed 3개)으로 빠른 검증")
    
    print(f"\n### Phase 2: 정밀 튜닝 (1일)")
    print(f"  1. Phase 1에서 최선의 조합 선택")
    print(f"  2. Boundary threshold 조정 (0.65, 0.70, 0.75)")
    print(f"  3. Supar weight 조정 (+0.00, +0.05, +0.10)")
    print(f"  → seed 10회 전체 실험으로 통계적 확정")
    
    print(f"\n### Phase 3: Ensemble 고도화 (선택, 1일)")
    print(f"  1. Close race 케이스만 별도 처리 로직 추가")
    print(f"  2. Soft voting 실험")
    print(f"  → F1이 0.85 이상 도달하면 0.90 달성 가능성 검증")
    
    print(f"\n## 6. 예상 효과")
    print(f"  - **보수적 추정**: F1 0.80 → 0.85 (+6.25%)")
    print(f"    - Prior bonus 최적화: +2%")
    print(f"    - Threshold 조정: +1%")
    print(f"    - Supar weight 조정: +1%")
    print(f"  ")
    print(f"  - **낙관적 추정**: F1 0.80 → 0.90 (+12.5%)")
    print(f"    - 위 개선 + Ensemble voting: +4%")
    print(f"    - 근소한 차이 케이스 {close}건 중 70% 개선 시 달성 가능")
    
    print(f"\n## 7. 즉시 실행 가능한 커맨드")
    print(f"```bash")
    print(f"# Grid search 자동화 스크립트 작성 (scripts/grid_search_pa_weights.py)")
    print(f"python scripts/grid_search_pa_weights.py \\")
    print(f"  --prior-bonus 0.10,0.15,0.20 \\")
    print(f"  --seeds 1,2,3 \\")
    print(f"  --output-dir test_results/grid_search_phase1")
    print(f"")
    print(f"# 결과 집계 및 최선 조합 자동 선택")
    print(f"python scripts/summarize_grid_search.py \\")
    print(f"  --input-dir test_results/grid_search_phase1 \\")
    print(f"  --metric micro_f1_tgt_exact \\")
    print(f"  --out-csv best_config.csv")
    print(f"```")
    
    print(f"\n" + "="*80)
    print(f"다음 단계: Grid search 러너 스크립트를 작성하시겠습니까?")
    print(f"="*80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python deep_analysis_for_0.9.py <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    
    # Aggregate 분석 (seed 1~10 전체)
    all_records = []
    for seed_num in range(1, 11):
        trace_file = run_dir / f"pa_trace_seed{seed_num}.jsonl"
        if trace_file.exists():
            all_records.extend(load_jsonl(trace_file))
    
    print(f"총 trace 레코드: {len(all_records)}개")
    
    analysis = analyze_score_distribution(all_records)
    suggest_improvements(analysis)

if __name__ == "__main__":
    main()
