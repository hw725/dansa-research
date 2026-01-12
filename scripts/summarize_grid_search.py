"""
Grid Search 결과 집계 스크립트
각 조합별 F1 평균/표준편차를 계산하고 최선 조합을 선택합니다.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import statistics

def load_results(grid_search_dir: Path) -> List[Dict]:
    """모든 실험 결과 로드"""
    results = []
    
    # 먼저 root의 summary.json 확인
    root_summary = grid_search_dir / "summary.json"
    if root_summary.exists():
        try:
            with open(root_summary, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 전체 요약 파일인 경우 results 리스트 반환
                if 'results' in data and isinstance(data['results'], list):
                    return data['results']
        except Exception as e:
            print(f"Warning: Failed to load {root_summary}: {e}")
    
    # 개별 summary.json 파일들 찾기
    for summary_file in grid_search_dir.glob("**/summary.json"):
        if summary_file == root_summary:
            continue  # 이미 처리함
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {summary_file}: {e}")
    
    return results

def group_by_config(results: List[Dict]) -> Dict:
    """설정별로 결과 그룹화"""
    grouped: Dict[str, Dict[str, Any]] = {}
    
    for result in results:
        config = result.get('config', {})
        # NOTE: 이전 구현은 일부 키만으로 그룹 키를 만들어
        # pa_selection_params 등 추가 레버가 반영되지 않고 서로 다른 설정이 한 그룹으로 합쳐지는 문제가 있었다.
        # config 전체를 안정적으로 직렬화해 키로 사용한다.
        config_key = json.dumps(config, sort_keys=True, ensure_ascii=False)

        if config_key not in grouped:
            grouped[config_key] = {"config": config, "seed_results": []}
        
        # 각 seed의 F1 점수
        for seed_result in result.get('seed_results', []):
            success = bool(seed_result.get('success', True))
            f1 = seed_result.get('micro_f1_tgt_exact', 0.0)
            grouped[config_key]["seed_results"].append({
                'seed': seed_result.get('seed'),
                'f1': f1,
                'similarity': seed_result.get('mean_similarity', 0.0),
                'success': success,
            })
    
    return grouped

def calculate_statistics(grouped: Dict) -> List[Dict]:
    """각 설정별 통계 계산"""
    stats = []
    
    for _config_key, payload in grouped.items():
        config = payload.get("config", {}) or {}
        seed_results = payload.get("seed_results", []) or []

        total_seeds = len(seed_results)
        success_results = [r for r in seed_results if bool(r.get('success', True))]
        failed_seeds = total_seeds - len(success_results)

        f1_scores = [r['f1'] for r in success_results]
        sim_scores = [r['similarity'] for r in success_results]

        # 모든 seed가 실패한 경우(출력 파일 미생성 등): 랭킹/평균 계산에서 제외
        if not f1_scores:
            continue
        
        stats.append({
            'config': config,
            'n_seeds': len(f1_scores),
            'n_seeds_total': total_seeds,
            'n_failed': failed_seeds,
            'f1_mean': statistics.mean(f1_scores),
            'f1_stdev': statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
            'f1_min': min(f1_scores),
            'f1_max': max(f1_scores),
            'sim_mean': statistics.mean(sim_scores),
            'seed_results': seed_results
        })
    
    # F1 평균 기준으로 내림차순 정렬
    stats.sort(key=lambda x: x['f1_mean'], reverse=True)
    
    return stats

def print_summary(stats: List[Dict], top_k: int = 10):
    """결과 요약 출력"""
    print("=" * 80)
    print("Grid Search 결과 요약")
    print("=" * 80)
    print(f"총 실험 조합 수: {len(stats)}")
    print()
    
    print(f"Top {min(top_k, len(stats))} 설정:")
    print("-" * 80)
    print(
        f"{'Rank':<5} {'PriorBonus':<12} {'BdryThres':<12} {'SuparBonus':<12} "
        f"{'Extra':<28} {'F1 Mean':<10} {'F1 Std':<10} {'Seeds':<6}"
    )
    print("-" * 80)
    
    for i, stat in enumerate(stats[:top_k], 1):
        config = stat['config']
        pb = config.get('prior_bonus', 0.0)
        bt = config.get('boundary_threshold', 0.0)
        sb = config.get('supar_bonus', 0.0)

        tuned = (config.get('_tuned') or {}).get('pa_selection_params') or {}
        extra_parts: List[str] = []
        if 'boundary_aware_weight' in tuned:
            extra_parts.append(f"bw={float(tuned['boundary_aware_weight']):.2f}")
        style = tuned.get('boundary_style_prior') or {}
        if isinstance(style, dict):
            if style.get('weight_terminal') is not None:
                extra_parts.append(f"st={float(style['weight_terminal']):.3f}")
            if style.get('weight_continuation') is not None:
                extra_parts.append(f"sc={float(style['weight_continuation']):.3f}")
        if 'max_candidates_multiplier' in tuned:
            extra_parts.append(f"mc={int(tuned['max_candidates_multiplier'])}")
        if 'penalty_empty_src' in tuned:
            extra_parts.append(f"pe={float(tuned['penalty_empty_src']):.2f}")
        psp = tuned.get('penalty_short_pairs') or {}
        if isinstance(psp, dict) and psp.get('penalty_per_pair') is not None:
            extra_parts.append(f"psp={float(psp['penalty_per_pair']):.3f}")

        extra = ",".join(extra_parts)[:28]

        print(
            f"{i:<5} {pb:<12.2f} {bt:<12.2f} {sb:<12.2f} "
            f"{extra:<28} {stat['f1_mean']:<10.4f} {stat['f1_stdev']:<10.4f} {stat['n_seeds']:<6}"
        )
    
    print("-" * 80)
    print()
    
    # 최고 성능 설정 상세 출력
    best = stats[0]
    print("🏆 최고 성능 설정:")
    print(json.dumps(best['config'], indent=2))
    print()
    print(f"평균 F1: {best['f1_mean']:.4f} (±{best['f1_stdev']:.4f})")
    print(f"F1 범위: [{best['f1_min']:.4f}, {best['f1_max']:.4f}]")
    print(f"평균 Similarity: {best['sim_mean']:.4f}")
    print(f"실험 횟수: {best['n_seeds']} seeds")
    print()

def save_results(stats: List[Dict], output_file: Path):
    """결과를 JSON 파일로 저장"""
    output_data = {
        'total_configs': len(stats),
        'best_config': stats[0]['config'] if stats else None,
        'best_f1': stats[0]['f1_mean'] if stats else None,
        'all_results': stats
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 전체 결과 저장: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Grid Search 결과 집계')
    parser.add_argument('grid_search_dir', type=str, help='Grid search 결과 디렉토리')
    parser.add_argument('--top-k', type=int, default=10, help='출력할 상위 조합 수 (기본: 10)')
    parser.add_argument('--output', type=str, help='결과 JSON 저장 경로 (선택)')
    
    args = parser.parse_args()
    
    grid_search_dir = Path(args.grid_search_dir)
    if not grid_search_dir.exists():
        print(f"❌ 디렉토리가 존재하지 않습니다: {grid_search_dir}")
        return
    
    print(f"📂 Grid Search 디렉토리: {grid_search_dir}")
    print()
    
    # 결과 로드
    results = load_results(grid_search_dir)
    if not results:
        print("❌ 결과 파일을 찾을 수 없습니다.")
        return
    
    print(f"✅ {len(results)} 개의 실험 결과 로드 완료")
    print()
    
    # 설정별 그룹화
    grouped = group_by_config(results)
    
    # 통계 계산
    stats = calculate_statistics(grouped)
    
    # 요약 출력
    print_summary(stats, top_k=args.top_k)
    
    # 결과 저장
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = grid_search_dir / "summary_aggregated.json"
    
    save_results(stats, output_file)
    
    print()
    print("=" * 80)
    print("다음 단계:")
    print(f"1. 최고 성능 설정을 csp_config.json에 반영")
    print(f"2. Phase 2 실행 (boundary_threshold, supar_bonus 추가 튜닝)")
    print("=" * 80)

if __name__ == "__main__":
    main()
