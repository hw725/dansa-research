#!/usr/bin/env python3
"""
PA 오류 케이스 심층 분석 스크립트
목적: F1 0.80 → 0.9 달성을 위한 개선점 파악
- 각 family별 정확도 분석
- 오류가 많이 발생하는 패턴 추출
- Best candidate 선택의 적절성 검증
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

def load_trace(trace_path: Path) -> List[dict]:
    """JSONL trace 파일 로드"""
    records = []
    with open(trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def load_results(result_csv: Path) -> Dict[str, dict]:
    """PA 결과 CSV 로드 (src_id를 키로)"""
    results = {}
    with open(result_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_id = row.get('src_id', row.get('para_id', ''))
            results[src_id] = row
    return results

def load_gold(gold_csv: Path) -> Dict[str, dict]:
    """Gold 정답 CSV 로드"""
    gold = {}
    with open(gold_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_id = row.get('src_id', row.get('para_id', ''))
            gold[src_id] = row
    return gold

def analyze_selection_stage(trace_records: List[dict], results: Dict[str, dict], gold: Dict[str, dict]):
    """선택 단계 분석: family별 성공률, 오류 패턴"""
    
    # Family별 통계
    family_stats = defaultdict(lambda: {
        'count': 0,
        'success': 0,
        'fail': 0,
        'scores': [],
        'error_patterns': Counter()
    })
    
    # 전체 케이스 분석
    for rec in trace_records:
        if rec.get('stage') != 'src_matched_selected':
            continue
        
        src_id = rec.get('src_id', '')
        best_tag = rec.get('meta', {}).get('best_tag', '')
        
        # Family 추출
        family = 'unknown'
        if best_tag.startswith('boundary'):
            family = 'boundary'
        elif best_tag.startswith('supar'):
            family = 'supar'
        elif best_tag.startswith('whitespace_dp'):
            family = 'whitespace_dp'
        elif best_tag.startswith('fallback'):
            family = 'fallback'
        
        family_stats[family]['count'] += 1
        
        # 성공/실패 판정 (간단하게 results에 있고 gold와 비교)
        if src_id in results and src_id in gold:
            # 실제로는 더 정교한 비교 필요하지만, 여기서는 예시로
            result_tgt = results[src_id].get('tgt_para_id', '')
            gold_tgt = gold[src_id].get('tgt_para_id', '')
            
            if result_tgt == gold_tgt:
                family_stats[family]['success'] += 1
            else:
                family_stats[family]['fail'] += 1
                
                # 오류 패턴 기록
                meta = rec.get('meta', {})
                candidates_total = meta.get('candidates_total', 0)
                candidates_considered = meta.get('candidates_considered', 0)
                
                pattern = f"total={candidates_total},considered={candidates_considered}"
                family_stats[family]['error_patterns'][pattern] += 1
        
        # 점수 기록
        best_score = rec.get('meta', {}).get('best_score', 0)
        family_stats[family]['scores'].append(best_score)
    
    return family_stats

def analyze_candidate_quality(trace_records: List[dict]):
    """후보 품질 분석: 최선의 선택이 맞았는가?"""
    
    analysis = {
        'total_cases': 0,
        'single_candidate_forced': 0,  # 후보가 1개뿐
        'score_margin_small': 0,  # 1등과 2등 차이 < 0.05
        'score_margin_medium': 0,  # 0.05 <= 차이 < 0.15
        'score_margin_large': 0,  # 차이 >= 0.15
        'boundary_vs_supar': {'boundary_win': 0, 'supar_win': 0, 'margin': []},
        'boundary_vs_whitespace': {'boundary_win': 0, 'whitespace_win': 0, 'margin': []},
        'supar_vs_whitespace': {'supar_win': 0, 'whitespace_win': 0, 'margin': []}
    }
    
    for rec in trace_records:
        if rec.get('stage') != 'src_matched_selected':
            continue
        
        meta = rec.get('meta', {})
        candidates = meta.get('top_candidates', [])
        
        if not candidates:
            continue
        
        analysis['total_cases'] += 1
        
        # 후보가 1개뿐인 케이스 (considered==1은 아니지만 점수 차이 확인)
        considered = [c for c in candidates if c.get('considered', False)]
        if len(considered) <= 1:
            analysis['single_candidate_forced'] += 1
            continue
        
        # 점수 기준 정렬
        sorted_cands = sorted(considered, key=lambda c: c.get('score', 0), reverse=True)
        
        if len(sorted_cands) >= 2:
            best = sorted_cands[0]
            second = sorted_cands[1]
            margin = best.get('score', 0) - second.get('score', 0)
            
            if margin < 0.05:
                analysis['score_margin_small'] += 1
            elif margin < 0.15:
                analysis['score_margin_medium'] += 1
            else:
                analysis['score_margin_large'] += 1
            
            # Family 간 비교
            best_fam = best.get('family', '')
            second_fam = second.get('family', '')
            
            pair = tuple(sorted([best_fam, second_fam]))
            if pair == ('boundary', 'supar'):
                if best_fam == 'boundary':
                    analysis['boundary_vs_supar']['boundary_win'] += 1
                else:
                    analysis['boundary_vs_supar']['supar_win'] += 1
                analysis['boundary_vs_supar']['margin'].append(margin)
            elif pair == ('boundary', 'whitespace_dp'):
                if best_fam == 'boundary':
                    analysis['boundary_vs_whitespace']['boundary_win'] += 1
                else:
                    analysis['boundary_vs_whitespace']['whitespace_win'] += 1
                analysis['boundary_vs_whitespace']['margin'].append(margin)
            elif pair == ('supar', 'whitespace_dp'):
                if best_fam == 'supar':
                    analysis['supar_vs_whitespace']['supar_win'] += 1
                else:
                    analysis['supar_vs_whitespace']['whitespace_win'] += 1
                analysis['supar_vs_whitespace']['margin'].append(margin)
    
    return analysis

def print_family_stats(family_stats: dict):
    """Family별 통계 출력"""
    print("\n=== Family별 성능 분석 ===")
    print(f"{'Family':<20} {'Count':<8} {'Success':<8} {'Fail':<8} {'Success%':<10} {'Avg Score':<10}")
    print("-" * 80)
    
    for family in sorted(family_stats.keys()):
        stats = family_stats[family]
        count = stats['count']
        success = stats['success']
        fail = stats['fail']
        success_rate = (success / (success + fail) * 100) if (success + fail) > 0 else 0
        avg_score = statistics.mean(stats['scores']) if stats['scores'] else 0
        
        print(f"{family:<20} {count:<8} {success:<8} {fail:<8} {success_rate:<10.2f} {avg_score:<10.4f}")
    
    print("\n=== Family별 주요 오류 패턴 ===")
    for family in sorted(family_stats.keys()):
        stats = family_stats[family]
        if stats['error_patterns']:
            print(f"\n{family}:")
            for pattern, cnt in stats['error_patterns'].most_common(3):
                print(f"  {pattern}: {cnt}건")

def print_candidate_quality(analysis: dict):
    """후보 품질 분석 결과 출력"""
    print("\n=== 후보 선택 품질 분석 ===")
    total = analysis['total_cases']
    print(f"총 케이스: {total}")
    print(f"단일 후보 강제: {analysis['single_candidate_forced']} ({analysis['single_candidate_forced']/total*100:.1f}%)")
    print(f"\n점수 차이 분포:")
    print(f"  작음 (<0.05):    {analysis['score_margin_small']} ({analysis['score_margin_small']/total*100:.1f}%)")
    print(f"  중간 (0.05~0.15): {analysis['score_margin_medium']} ({analysis['score_margin_medium']/total*100:.1f}%)")
    print(f"  큼 (>=0.15):     {analysis['score_margin_large']} ({analysis['score_margin_large']/total*100:.1f}%)")
    
    print(f"\n=== Family 간 경쟁 분석 ===")
    
    # boundary vs supar
    bvs = analysis['boundary_vs_supar']
    if bvs['boundary_win'] + bvs['supar_win'] > 0:
        print(f"\nBoundary vs Supar:")
        print(f"  Boundary 승: {bvs['boundary_win']}")
        print(f"  Supar 승: {bvs['supar_win']}")
        if bvs['margin']:
            print(f"  평균 점수차: {statistics.mean(bvs['margin']):.4f}")
    
    # boundary vs whitespace
    bvw = analysis['boundary_vs_whitespace']
    if bvw['boundary_win'] + bvw['whitespace_win'] > 0:
        print(f"\nBoundary vs Whitespace_DP:")
        print(f"  Boundary 승: {bvw['boundary_win']}")
        print(f"  Whitespace 승: {bvw['whitespace_win']}")
        if bvw['margin']:
            print(f"  평균 점수차: {statistics.mean(bvw['margin']):.4f}")
    
    # supar vs whitespace
    svw = analysis['supar_vs_whitespace']
    if svw['supar_win'] + svw['whitespace_win'] > 0:
        print(f"\nSupar vs Whitespace_DP:")
        print(f"  Supar 승: {svw['supar_win']}")
        print(f"  Whitespace 승: {svw['whitespace_win']}")
        if svw['margin']:
            print(f"  평균 점수차: {statistics.mean(svw['margin']):.4f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_pa_errors.py <run_dir>")
        print("  run_dir: 실험 결과가 있는 디렉토리 (예: test_results/multitest_seed1_10_markerbonusA_skipfixA/20260106_074514)")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)
    
    print(f"분석 대상: {run_dir}")
    
    # Seed별 분석 (예: seed1만 상세 분석)
    for seed_num in [1, 2, 3]:  # 일단 3개 seed만
        trace_file = run_dir / f"pa_trace_seed{seed_num}.jsonl"
        result_file = run_dir / f"pa_output_n100_seed{seed_num}.csv"
        gold_file = run_dir / f"pa_gold_subset_n100_seed{seed_num}.csv"
        
        if not trace_file.exists():
            print(f"Trace 파일 없음: {trace_file}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Seed {seed_num} 분석")
        print(f"{'='*80}")
        
        trace = load_trace(trace_file)
        
        # 결과/정답 로드 (있으면)
        results = load_results(result_file) if result_file.exists() else {}
        gold = load_gold(gold_file) if gold_file.exists() else {}
        
        # Family별 분석
        family_stats = analyze_selection_stage(trace, results, gold)
        print_family_stats(family_stats)
        
        # 후보 품질 분석
        candidate_quality = analyze_candidate_quality(trace)
        print_candidate_quality(candidate_quality)

if __name__ == "__main__":
    main()
