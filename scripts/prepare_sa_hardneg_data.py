#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SA Hard Negative 데이터 생성

- 기존 SA boundary 모델로 추론 → 오탐/미탐 케이스 수집
- 어려운 경계 케이스에 가중치 부여
- 출력: datasets/sa_boundary_hardneg_v2/

관측성 우선 원칙:
- 모든 처리 단계를 JSONL로 trace
- 오류 케이스별 통계 출력
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import torch

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"


def load_jsonl(path: Path) -> List[Dict]:
    """JSONL 파일 로드"""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: List[Dict], path: Path):
    """JSONL 파일 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"💾 Saved: {path} ({len(samples)} samples)")


def load_boundary_model(model_path: Path, device: str = "cuda"):
    """Boundary 모델 로드 (간단 버전)"""
    from train_boundary_multitask import MultiHeadBoundary, encode_text
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    max_len = ckpt.get("max_len", 1024)
    tasks = ckpt.get("tasks", ["pa", "sa", "pd"])
    
    model = MultiHeadBoundary(
        vocab_size=len(vocab) + 1,
        hidden_dim=128,
        emb_dim=64,
        tasks=tasks
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, vocab, max_len


def predict_boundaries(
    model,
    text: str,
    vocab: Dict[str, int],
    max_len: int,
    device: str = "cuda",
    threshold: float = 0.5,
) -> List[int]:
    """경계 예측 (B 위치 리스트 반환)"""
    # 텍스트 인코딩
    ids = [vocab.get(ch, 0) for ch in text][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits = model(x, "sa")
        probs = torch.sigmoid(logits[0]).cpu().numpy()
    
    # threshold 이상인 위치를 경계로 예측
    pred_boundaries = []
    for i, p in enumerate(probs[:len(text)]):
        if p >= threshold:
            pred_boundaries.append(i)
    
    return pred_boundaries


def compute_errors(
    gold_labels: str,
    pred_boundaries: List[int],
) -> Tuple[List[int], List[int], List[int]]:
    """오탐/미탐 케이스 계산
    
    Returns:
        (true_positives, false_positives, false_negatives)
    """
    # Gold 경계 위치 추출
    gold_boundaries = set()
    for i, ch in enumerate(gold_labels):
        if ch == "B":
            gold_boundaries.add(i)
    
    pred_set = set(pred_boundaries)
    
    tp = list(gold_boundaries & pred_set)
    fp = list(pred_set - gold_boundaries)  # 오탐: 예측했지만 실제 아님
    fn = list(gold_boundaries - pred_set)  # 미탐: 실제 경계인데 놓침
    
    return tp, fp, fn


def generate_hardneg_sample(
    sample: Dict,
    fp_positions: List[int],
    fn_positions: List[int],
    hardneg_weight: float = 1.5,
) -> Dict:
    """Hard Negative 가중치가 적용된 샘플 생성"""
    new_sample = sample.copy()
    
    # 가중치 배열 생성 (기본 1.0)
    text_len = len(sample["text"])
    weights = [1.0] * text_len
    
    # 오탐/미탐 위치에 가중치 부여
    for pos in fp_positions:
        if 0 <= pos < text_len:
            weights[pos] = hardneg_weight
    
    for pos in fn_positions:
        if 0 <= pos < text_len:
            weights[pos] = hardneg_weight
    
    # 가중치 문자열로 저장 (JSON 직렬화용)
    new_sample["weights"] = weights
    new_sample["fp_count"] = len(fp_positions)
    new_sample["fn_count"] = len(fn_positions)
    new_sample["is_hardneg"] = len(fp_positions) > 0 or len(fn_positions) > 0
    
    return new_sample


def prepare_sa_hardneg(
    model_path: str = None,
    input_dir: str = None,
    output_dir: str = None,
    threshold: float = 0.5,
    hardneg_weight: float = 1.5,
    device: str = "cuda",
    trace_path: str = None,
):
    """SA Hard Negative 데이터 생성 메인 함수"""
    
    # 기본 경로 설정
    if model_path is None:
        model_path = MODELS_ROOT / "boundary_multitask.pt"
    else:
        model_path = Path(model_path)
    
    if input_dir is None:
        input_dir = DATASETS_ROOT / "sa_boundary"
    else:
        input_dir = Path(input_dir)
    
    if output_dir is None:
        output_dir = DATASETS_ROOT / "sa_boundary_hardneg_v2"
    else:
        output_dir = Path(output_dir)
    
    print(f"📂 Input: {input_dir}")
    print(f"📂 Output: {output_dir}")
    print(f"🔧 Model: {model_path}")
    print(f"⚙️ Threshold: {threshold}, HardNeg Weight: {hardneg_weight}")
    
    # 모델 로드
    device = device if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")
    
    model, vocab, max_len = load_boundary_model(model_path, device)
    print(f"✅ Model loaded (vocab_size={len(vocab)}, max_len={max_len})")
    
    # Trace 초기화
    trace_records = []
    
    # 각 split 처리
    for split in ["train", "val", "test"]:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            print(f"⚠️ {split}.jsonl not found, skipping")
            continue
        
        samples = load_jsonl(input_path)
        print(f"\n📊 Processing {split}: {len(samples)} samples")
        
        # 통계
        stats = {
            "total": len(samples),
            "hardneg_count": 0,
            "total_fp": 0,
            "total_fn": 0,
            "total_tp": 0,
        }
        
        output_samples = []
        
        for i, sample in enumerate(samples):
            text = sample.get("text", "")
            labels = sample.get("labels", "")
            
            if not text or not labels:
                output_samples.append(sample)
                continue
            
            # 경계 예측
            pred_boundaries = predict_boundaries(
                model, text, vocab, max_len, device, threshold
            )
            
            # 오류 계산
            tp, fp, fn = compute_errors(labels, pred_boundaries)
            
            stats["total_tp"] += len(tp)
            stats["total_fp"] += len(fp)
            stats["total_fn"] += len(fn)
            
            # Hard Negative 샘플 생성
            new_sample = generate_hardneg_sample(
                sample, fp, fn, hardneg_weight
            )
            
            if new_sample["is_hardneg"]:
                stats["hardneg_count"] += 1
            
            output_samples.append(new_sample)
            
            # 진행률 표시
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1}/{len(samples)}...")
        
        # 결과 저장
        output_path = output_dir / f"{split}.jsonl"
        save_jsonl(output_samples, output_path)
        
        # 통계 출력
        precision = stats["total_tp"] / max(1, stats["total_tp"] + stats["total_fp"])
        recall = stats["total_tp"] / max(1, stats["total_tp"] + stats["total_fn"])
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        
        print(f"\n📈 {split} Statistics:")
        print(f"   Total samples: {stats['total']}")
        print(f"   HardNeg samples: {stats['hardneg_count']} ({100*stats['hardneg_count']/max(1,stats['total']):.1f}%)")
        print(f"   Total TP: {stats['total_tp']}")
        print(f"   Total FP (오탐): {stats['total_fp']}")
        print(f"   Total FN (미탐): {stats['total_fn']}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1: {f1:.4f}")
        
        # Trace 기록
        trace_records.append({
            "timestamp": datetime.now().isoformat(),
            "stage": "prepare_hardneg",
            "split": split,
            "stats": stats,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    
    # Trace 저장
    if trace_path:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as f:
            for r in trace_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n📝 Trace saved: {trace_path}")
    
    print("\n✅ SA Hard Negative 데이터 생성 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SA Hard Negative 데이터 생성")
    parser.add_argument("--model", type=str, default=None,
                        help="Boundary 모델 경로 (기본: models/boundary_multitask.pt)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="입력 디렉토리 (기본: datasets/sa_boundary)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="출력 디렉토리 (기본: datasets/sa_boundary_hardneg_v2)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="경계 예측 threshold (기본: 0.5)")
    parser.add_argument("--hardneg-weight", type=float, default=1.5,
                        help="Hard Negative 가중치 (기본: 1.5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="디바이스 (기본: cuda)")
    parser.add_argument("--trace", type=str, default=None,
                        help="Trace JSONL 출력 경로")
    
    args = parser.parse_args()
    
    prepare_sa_hardneg(
        model_path=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        hardneg_weight=args.hardneg_weight,
        device=args.device,
        trace_path=args.trace,
    )
