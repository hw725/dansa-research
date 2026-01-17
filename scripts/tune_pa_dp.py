"""PA DP 파라미터 베이지안 최적화 (Optuna) - 직접 호출 버전

Usage:
    docker exec -e CSP_BGE_CACHE_DIR=/workspace/test_results/_cache/bge_embeddings \
        csp-workspace python /workspace/scripts/tune_pa_dp.py \
        --n-trials 20 \
        --input /workspace/test_results/pa_test_input_30.xlsx \
        --gold /workspace/test_results/pa_gold_sample_30.csv \
        --output-dir /workspace/test_results/optuna_runs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pa"))
sys.path.insert(0, str(Path(__file__).parent.parent / "accuracy"))

import argparse
import json
import tempfile
import pandas as pd
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# PA 관련 import
from processor import process_paragraph_file
from pa_evaluator import _read_tabular, _boundary_positions_normed


def evaluate_pa_output(pred_path: Path, gold_path: Path) -> float:
    """PA 출력과 gold를 비교하여 tgt일치 subset의 원문 경계 F1 반환"""
    pred_df = _read_tabular(pred_path)
    gold_df = _read_tabular(gold_path)
    
    for col in ("원문", "번역문"):
        pred_df[col] = pred_df[col].fillna("")
        gold_df[col] = gold_df[col].fillna("")
    
    pred_df["문단식별자"] = pred_df["문단식별자"].astype(int)
    gold_df["문단식별자"] = gold_df["문단식별자"].astype(int)
    
    pred_has_book = "book_name" in pred_df.columns
    if pred_has_book:
        pred_df["book_name"] = pred_df["book_name"].fillna("").astype(str)
    gold_df["book_name"] = gold_df["book_name"].fillna("").astype(str)
    
    if pred_has_book:
        pred_groups = pred_df.groupby(["book_name", "문단식별자"], sort=False)
        gold_groups = gold_df.sort_values(["book_name", "문단식별자", "문장식별자"], kind="stable").groupby(
            ["book_name", "문단식별자"], sort=False
        )
        common_keys = sorted(set(pred_groups.groups.keys()) & set(gold_groups.groups.keys()))
    else:
        pred_groups = pred_df.groupby("문단식별자", sort=False)
        gold_groups = gold_df.sort_values(["문단식별자", "문장식별자"], kind="stable").groupby("문단식별자", sort=False)
        common_keys = sorted(set(pred_groups.groups.keys()) & set(gold_groups.groups.keys()))
    
    all_pred_bounds = set()
    all_gold_bounds = set()
    
    for key in common_keys:
        pred_rows = pred_groups.get_group(key)
        gold_rows = gold_groups.get_group(key)
        
        pred_tgt_list = pred_rows["번역문"].tolist()
        gold_tgt_list = gold_rows["번역문"].tolist()
        
        if pred_tgt_list != gold_tgt_list:
            continue
        
        pred_bounds = {(key, p) for p in _boundary_positions_normed(pred_rows["원문"].tolist())}
        gold_bounds = {(key, p) for p in _boundary_positions_normed(gold_rows["원문"].tolist())}
        
        all_pred_bounds.update(pred_bounds)
        all_gold_bounds.update(gold_bounds)
    
    if not all_gold_bounds:
        return 0.0
    
    tp = len(set(all_pred_bounds) & set(all_gold_bounds))
    prec = tp / len(all_pred_bounds) if all_pred_bounds else 0.0
    rec = tp / len(all_gold_bounds) if all_gold_bounds else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return f1


def objective(trial: optuna.Trial, input_path: str, gold_path: str, output_dir: Path) -> float:
    """Optuna objective function"""
    boundary_threshold = trial.suggest_float("boundary_threshold", 0.4, 0.8)
    boundary_bonus_factor = trial.suggest_float("boundary_bonus_factor", 0.5, 3.0)
    shift_penalty_factor = trial.suggest_float("shift_penalty_factor", 0.0001, 0.002, log=True)
    
    output_path = output_dir / f"trial_{trial.number:03d}.xlsx"
    
    try:
        process_paragraph_file(
            input_file=input_path,
            output_file=str(output_path),
            embedder_name="bge",
            max_length=180,
            similarity_threshold=0.7,
            max_workers=4,
            batch_size=256,
            verbose=False,
            device="cuda",
            use_boundary_model=True,
            boundary_threshold=boundary_threshold,
            enable_refine=True,
            enable_adjacent_boundary_refine=True,
            boundary_bonus_factor=boundary_bonus_factor,
            shift_penalty_factor=shift_penalty_factor,
        )
    except Exception as e:
        print(f"[Trial {trial.number}] Error: {e}")
        return 0.0
    
    if not output_path.exists():
        return 0.0
    
    f1 = evaluate_pa_output(output_path, Path(gold_path))
    
    print(f"[Trial {trial.number}] bt={boundary_threshold:.3f} bf={boundary_bonus_factor:.2f} sp={shift_penalty_factor:.5f} → F1={f1:.4f}")
    
    return f1


def main():
    parser = argparse.ArgumentParser(description="PA DP 파라미터 베이지안 최적화")
    parser.add_argument("--n-trials", type=int, default=20, help="시행 횟수")
    parser.add_argument("--input", required=True, help="입력 파일")
    parser.add_argument("--gold", required=True, help="정답 파일")
    parser.add_argument("--output-dir", required=True, help="출력 디렉토리")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔧 모델 초기화 중...")
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="pa_dp_tuning",
    )
    
    print(f"🚀 Optuna 최적화 시작 (n_trials={args.n_trials})")
    
    study.optimize(
        lambda trial: objective(trial, args.input, args.gold, output_dir),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    print("\n" + "="*60)
    print("최적화 완료!")
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    
    with open(output_dir / "optuna_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n결과 저장: {output_dir / 'optuna_results.json'}")


if __name__ == "__main__":
    main()
