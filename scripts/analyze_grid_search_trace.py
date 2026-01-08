#!/usr/bin/env python3
"""Grid Search trace 분석 스크립트

목적: Grid Search 실험에서 prior_bonus가 실제로 적용되었는지 확인
- 각 config의 trace 파일에서 prior_bonus 값 추출
- 후보 선택 통계 집계
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import sys


def analyze_trace_file(trace_path: Path) -> Dict[str, Any]:
    """단일 trace 파일 분석"""
    if not trace_path.exists():
        return {"error": "trace file not found"}

    prior_bonuses = []
    best_tags = []
    candidates_considered = []
    candidates_total = []

    with open(trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                stage = record.get("stage", "")

                if stage == "src_matched_selected":
                    meta = record.get("meta", {})

                    # 후보 통계
                    candidates_total.append(meta.get("candidates_total", 0))
                    candidates_considered.append(meta.get("candidates_considered", 0))

                    # Best 후보
                    best_tag = meta.get("best_tag", "")
                    if best_tag:
                        best_tags.append(best_tag)

                    # 모든 후보의 prior_bonus 수집
                    top_candidates = meta.get("top_candidates", [])
                    for cand in top_candidates:
                        pb = cand.get("prior_bonus", 0.0)
                        if pb != 0.0:  # 0이 아닌 것만 기록
                            prior_bonuses.append({
                                "tag": cand.get("tag", ""),
                                "prior_bonus": pb,
                                "score": cand.get("score", 0.0),
                                "avg_similarity": cand.get("avg_similarity", 0.0)
                            })
            except Exception as e:
                continue

    # 통계 집계
    result = {
        "total_paragraphs": len(best_tags),
        "prior_bonuses_found": len(prior_bonuses),
        "prior_bonus_values": {},
        "best_tag_distribution": {},
        "avg_candidates_total": sum(candidates_total) / len(candidates_total) if candidates_total else 0,
        "avg_candidates_considered": sum(candidates_considered) / len(candidates_considered) if candidates_considered else 0,
    }

    # prior_bonus 값 분포
    for pb_info in prior_bonuses:
        val = pb_info["prior_bonus"]
        if val not in result["prior_bonus_values"]:
            result["prior_bonus_values"][val] = 0
        result["prior_bonus_values"][val] += 1

    # 선택된 후보 분포
    for tag in best_tags:
        family = tag.split("(")[0] if "(" in tag else tag
        if family not in result["best_tag_distribution"]:
            result["best_tag_distribution"][family] = 0
        result["best_tag_distribution"][family] += 1

    # 샘플 prior_bonus (처음 5개)
    result["prior_bonus_samples"] = prior_bonuses[:5]

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_grid_search_trace.py <grid_search_output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Grid Search Trace 분석: {output_dir}")
    print(f"{'='*80}\n")

    # 모든 config 디렉토리 찾기
    config_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('pb')])

    if not config_dirs:
        print("Error: No config directories found (expecting pb*)")
        sys.exit(1)

    results_by_config = {}

    for config_dir in config_dirs:
        config_name = config_dir.name
        print(f"\n## Config: {config_name}")
        print("-" * 80)

        # 각 seed의 trace 파일 분석
        seed_results = {}
        for seed_dir in sorted(config_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith('seed'):
                continue

            seed = seed_dir.name
            trace_file = seed_dir / f"pa_trace_{seed}.jsonl"

            result = analyze_trace_file(trace_file)
            seed_results[seed] = result

            if "error" in result:
                print(f"  {seed}: {result['error']}")
                continue

            print(f"\n  {seed}:")
            print(f"    Total paragraphs: {result['total_paragraphs']}")
            print(f"    Prior bonuses found: {result['prior_bonuses_found']}")

            if result['prior_bonus_values']:
                print(f"    Prior bonus values:")
                for val, count in sorted(result['prior_bonus_values'].items()):
                    print(f"      {val}: {count} times")
            else:
                print(f"    [WARNING] No non-zero prior_bonus values found!")

            print(f"    Best tag distribution:")
            for family, count in result['best_tag_distribution'].items():
                pct = count / result['total_paragraphs'] * 100 if result['total_paragraphs'] > 0 else 0
                print(f"      {family}: {count} ({pct:.1f}%)")

            print(f"    Avg candidates: total={result['avg_candidates_total']:.1f}, considered={result['avg_candidates_considered']:.1f}")

            if result.get('prior_bonus_samples'):
                print(f"    Sample prior_bonus entries (first 3):")
                for i, sample in enumerate(result['prior_bonus_samples'][:3]):
                    print(f"      [{i+1}] {sample['tag']}: prior_bonus={sample['prior_bonus']}, score={sample['score']:.4f}")

        results_by_config[config_name] = seed_results

    # 요약 비교
    print(f"\n{'='*80}")
    print("Summary: Config-wise prior_bonus verification")
    print(f"{'='*80}\n")

    print(f"{'Config':<20} {'Seed':<10} {'PB Found':<12} {'PB Values':<30}")
    print("-" * 80)

    for config_name in sorted(results_by_config.keys()):
        for seed, result in sorted(results_by_config[config_name].items()):
            if "error" in result:
                continue

            pb_found = result.get('prior_bonuses_found', 0)
            pb_vals = ", ".join([f"{v:.3f}" for v in sorted(result.get('prior_bonus_values', {}).keys())])

            print(f"{config_name:<20} {seed:<10} {pb_found:<12} {pb_vals:<30}")

    print(f"\n{'='*80}")
    print("[OK] Analysis completed")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
