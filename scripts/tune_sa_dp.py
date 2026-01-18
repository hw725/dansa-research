#!/usr/bin/env python3
"""SA DP 파라미터 랜덤 서치 최적화

Usage:
    python scripts/tune_sa_dp.py --n-trials 10 --sample-size 50

최적화 대상:
    - dp_window, boundary_bonus, particle_bonus, length_penalty, sim_gamma
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from sa.io_manager import process_file


# 파라미터 범위
PARAM_RANGES = {
    "dp_window": [2, 3, 4, 5],
    "boundary_bonus": (0.1, 0.5),
    "particle_bonus": (0.1, 0.5),
    "length_penalty": (0.02, 0.15),
    "sim_gamma": (0.8, 1.5),
}


def sample_params(seed: int = None) -> dict:
    """랜덤 파라미터 샘플링"""
    if seed is not None:
        random.seed(seed)
    return {
        "dp_window": random.choice(PARAM_RANGES["dp_window"]),
        "boundary_bonus": round(random.uniform(*PARAM_RANGES["boundary_bonus"]), 2),
        "particle_bonus": round(random.uniform(*PARAM_RANGES["particle_bonus"]), 2),
        "length_penalty": round(random.uniform(*PARAM_RANGES["length_penalty"]), 3),
        "sim_gamma": round(random.uniform(*PARAM_RANGES["sim_gamma"]), 2),
    }


def evaluate(pred_df: pd.DataFrame, gold_df: pd.DataFrame, sent_ids: list) -> dict:
    """SA 출력과 Gold 비교"""
    gold_counts = gold_df.groupby('문장식별자').size()
    pred_counts = pred_df.groupby('문장식별자').size()
    
    exact_match = sum(1 for s in sent_ids if gold_counts.get(s, 0) == pred_counts.get(s, 0))
    segment_match_rate = exact_match / len(sent_ids) if sent_ids else 0
    
    # F1 근사 (set-based)
    tp, fp, fn = 0, 0, 0
    for sent_id in sent_ids:
        gold_segs = set(str(s).strip() for s in gold_df[gold_df['문장식별자'] == sent_id]['번역문'])
        pred_segs = set(str(s).strip() for s in pred_df[pred_df['문장식별자'] == sent_id]['번역문'])
        tp += len(gold_segs & pred_segs)
        fp += len(pred_segs - gold_segs)
        fn += len(gold_segs - pred_segs)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    return {"segment_match_rate": segment_match_rate, "f1": f1, "precision": prec, "recall": rec}


def run_trial(params: dict, input_df: pd.DataFrame, gold_df: pd.DataFrame,
              sent_ids: list, trial_num: int, output_dir: Path) -> dict:
    """단일 시행 실행"""
    temp_input = output_dir / f"trial_{trial_num:03d}_input.xlsx"
    temp_output = output_dir / f"trial_{trial_num:03d}_output.xlsx"
    
    input_df.to_excel(temp_input, index=False)
    
    try:
        success = process_file(
            input_file=str(temp_input),
            output_file=str(temp_output),
            embedder_name='bge',
            max_workers=4,
            chunk_size=50,
            use_parallel=True,
            verbose=False,
            **params
        )
        
        if not success or not temp_output.exists():
            return {"params": params, "error": "SA 실패", "f1": 0}
        
        pred_df = pd.read_excel(temp_output)
        metrics = evaluate(pred_df, gold_df, sent_ids)
        
        return {"params": params, **metrics}
        
    except Exception as e:
        return {"params": params, "error": str(e), "f1": 0}
    finally:
        try:
            temp_input.unlink()
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="SA DP 파라미터 랜덤 서치")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="test_results/sa_tuning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔧 SA DP 파라미터 탐색")
    print("=" * 60)
    
    # 데이터 로드
    input_df = pd.read_csv("datasets/pa/test.csv")
    gold_df = pd.read_csv("datasets/sa/test.csv")
    
    unique_sent_ids = list(input_df['문장식별자'].unique()[:args.sample_size])
    sample_df = input_df[input_df['문장식별자'].isin(unique_sent_ids)].copy()
    
    print(f"📂 샘플: {len(unique_sent_ids)}문장, {len(sample_df)}행")
    print(f"🎯 시행: {args.n_trials}회\n")
    
    results = []
    best_f1 = 0
    best_params = None
    
    for i in range(args.n_trials):
        params = sample_params()
        print(f"[{i+1}/{args.n_trials}] {params}")
        
        result = run_trial(params, sample_df, gold_df, unique_sent_ids, i, output_dir)
        results.append(result)
        
        if result.get("f1", 0) > best_f1:
            best_f1 = result["f1"]
            best_params = params
        
        print(f"         → F1={result.get('f1', 0):.4f} SMR={result.get('segment_match_rate', 0):.2%}")
    
    print("\n" + "=" * 60)
    print(f"🎉 Best F1: {best_f1:.4f}")
    print(f"🏆 Best params: {best_params}")
    
    # 결과 저장
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"best_f1": best_f1, "best_params": best_params, "trials": results,
                   "timestamp": datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)
    
    return 0


if __name__ == "__main__":
    exit(main())
