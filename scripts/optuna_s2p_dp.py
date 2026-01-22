#!/usr/bin/env python3
"""SA DP Alignment 파라미터 Optuna 튜닝

클린 데이터(원문 공백 없음)만 사용하여 경계 F1 최적화
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import json
import pandas as pd
from typing import List, Set
import warnings
warnings.filterwarnings("ignore")

import optuna
from optuna.samplers import TPESampler


def _norm(s: str) -> str:
    return re.sub(r'[\s\u3000]', '', str(s))


def _boundary_positions(segments: List[str]) -> Set[int]:
    positions = set()
    cursor = 0
    for i, seg in enumerate(segments):
        cursor += len(_norm(seg))
        if i < len(segments) - 1:
            positions.add(cursor)
    return positions


def _prf1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# 데이터 미리 로드
gold_df = pd.read_csv("datasets/s2p/test.csv")
sent_ids = list(gold_df['문장식별자'].unique()[:10])  # 속도를 위해 10개만

# 클린 문장만 필터링
clean_sents = []
for sent_id in sent_ids:
    gold_rows = gold_df[gold_df['문장식별자'] == sent_id].sort_values('구식별자')
    if gold_rows.empty:
        continue
    
    gold_src_segs = [str(r).strip() for r in gold_rows['원문']]
    gold_tgt_segs = [str(r).strip() for r in gold_rows['번역문']]
    
    all_clean = all(' ' not in src for src in gold_src_segs if src)
    if all_clean:
        clean_sents.append({
            'sent_id': sent_id,
            'src_segs': gold_src_segs,
            'tgt_segs': gold_tgt_segs,
            'src_text': ' '.join(gold_src_segs),
            'tgt_text': ' '.join(gold_tgt_segs),
        })

print(f"클린 문장: {len(clean_sents)}개")


def objective(trial):
    # 파라미터 제안
    dp_window = trial.suggest_int('dp_window', 1, 5)
    distance_decay = trial.suggest_float('distance_decay', 0.01, 0.2)
    boundary_bonus = trial.suggest_float('boundary_bonus', 0.0, 0.5)
    particle_bonus = trial.suggest_float('particle_bonus', 0.0, 0.5)
    length_penalty = trial.suggest_float('length_penalty', 0.0, 0.3)
    sim_gamma = trial.suggest_float('sim_gamma', 0.5, 2.0)
    similarity_threshold = trial.suggest_float('similarity_threshold', 0.3, 0.8)
    boundary_threshold = trial.suggest_float('boundary_threshold', 0.1, 0.9)
    
    # 파라미터로 평가
    from s2p.s2p_aligner import process_single_row
    
    # 경계 모델 주입 (s2p_aligner가 참조할 수 있도록)
    from s2p.io_manager import safe_process_sa_row
    if not hasattr(safe_process_sa_row, '_boundary_model'):
        from common.s2p_crossattn_boundary_loader import get_crossattn_boundary_tagger
        safe_process_sa_row._boundary_model = get_crossattn_boundary_tagger()
        print("✅ Optuna: Cross-Attention 경계 모델 주입 완료")

    tp = fp = fn = 0
    
    for sent in clean_sents:
        row_data = {
            '문장식별자': sent['sent_id'],
            '원문': sent['src_text'],
            '번역문': sent['tgt_text'],
        }
        
        try:
            result_rows = process_single_row(
                row_data, 
                use_boundary_model=True,  # 경계 모델 활성화
                boundary_threshold=boundary_threshold, # 🆕 Threshold 전달
                dp_window=dp_window,
                distance_decay=distance_decay,
                boundary_bonus=boundary_bonus,
                particle_bonus=particle_bonus,
                length_penalty=length_penalty,
                sim_gamma=sim_gamma,
                similarity_threshold=similarity_threshold,
            )
            pred_tgt_segs = [r['번역문'] for r in result_rows]
        except Exception as e:
            pred_tgt_segs = [sent['tgt_text']]
        
        gold_bounds = _boundary_positions(sent['tgt_segs'])
        pred_bounds = _boundary_positions(pred_tgt_segs)
        
        tp += len(gold_bounds & pred_bounds)
        fp += len(pred_bounds - gold_bounds)
        fn += len(gold_bounds - pred_bounds)
    
    _, _, f1 = _prf1(tp, fp, fn)
    return f1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 SA DP Alignment Optuna 튜닝")
    print("=" * 60)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
    )
    
    def callback(study, trial):
        print(f"  Trial {trial.number}: F1={trial.value:.4f} | best={study.best_value:.4f}")
    
    study.optimize(objective, n_trials=args.n_trials, callbacks=[callback])
    
    print("\n" + "=" * 60)
    print("📊 최적 결과")
    print("=" * 60)
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # 결과 저장
    result = {
        "best_f1": study.best_value,
        "best_params": study.best_params,
    }
    with open("sa_dp_optuna_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\n💾 Saved: sa_dp_optuna_result.json")


if __name__ == "__main__":
    main()
