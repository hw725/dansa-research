#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SA Head Fine-tuning (PA/PD Head 동결)

- 기존 boundary_multitask.pt 로드
- PA/PD head 파라미터 동결
- SA head만 추가 학습 (Hard Negative 가중치 적용)

PA 무영향 원칙:
- PA head는 절대 수정하지 않음
- 학습 후에도 PA 성능 보존 검증

관측성 우선 원칙:
- 모든 epoch의 loss/F1 기록
- 컴포넌트별 기여도 추적
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"


class BoundarySample:
    """경계 태깅 샘플"""
    def __init__(self, text: str, labels: str, task: str, weights: List[float] = None):
        self.text = text
        self.labels = labels
        self.task = task
        self.weights = weights if weights else [1.0] * len(text)


def load_jsonl(path: Path, task: str = "sa") -> List[BoundarySample]:
    """JSONL 파일 로드"""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            weights = obj.get("weights", None)
            samples.append(BoundarySample(
                obj["text"],
                obj["labels"],
                obj.get("task", task),
                weights
            ))
    return samples


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    """텍스트를 ID로 인코딩"""
    ids = [vocab.get(ch, 0) for ch in text][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def encode_labels(labels: str, max_len: int) -> torch.Tensor:
    """라벨을 binary로 인코딩"""
    arr = [1.0 if ch == "B" else 0.0 for ch in labels][:max_len]
    if len(arr) < max_len:
        arr += [0.0] * (max_len - len(arr))
    return torch.tensor(arr, dtype=torch.float)


def encode_weights(weights: List[float], max_len: int) -> torch.Tensor:
    """가중치를 텐서로 인코딩"""
    w = weights[:max_len]
    if len(w) < max_len:
        w = w + [1.0] * (max_len - len(w))
    return torch.tensor(w, dtype=torch.float)


class SAFinetuneDataset(Dataset):
    """SA Fine-tuning용 데이터셋"""
    def __init__(self, samples: List[BoundarySample], vocab: Dict[str, int], max_len: int):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        x = encode_text(s.text, self.vocab, self.max_len)
        y = encode_labels(s.labels, self.max_len)
        w = encode_weights(s.weights, self.max_len)
        return x, y, w


class MultiHeadBoundary(nn.Module):
    """멀티태스크 경계 태깅 모델"""
    def __init__(self, vocab_size: int, hidden_dim: int = 128, emb_dim: int = 64, tasks: List[str] = None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.heads = nn.ModuleDict({t: nn.Linear(hidden_dim * 2, 1) for t in tasks})
    
    def forward(self, x: torch.Tensor, task: str):
        h, _ = self.lstm(self.emb(x))
        logits = self.heads[task](h).squeeze(-1)
        return logits


def f1_from_probs(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float]:
    """Precision, Recall, F1 계산"""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    tp = (preds * labels).sum().item()
    fp = (preds * (1 - labels)).sum().item()
    fn = ((1 - preds) * labels).sum().item()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def evaluate(model: MultiHeadBoundary, loader: DataLoader, device: torch.device, task: str = "sa") -> Dict[str, float]:
    """모델 평가"""
    model.eval()
    total_p, total_r, total_f = 0.0, 0.0, 0.0
    count = 0
    
    with torch.no_grad():
        for x, y, w in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x, task)
            p, r, f = f1_from_probs(logits, y)
            total_p += p
            total_r += r
            total_f += f
            count += 1
    
    return {
        "p": total_p / max(1, count),
        "r": total_r / max(1, count),
        "f": total_f / max(1, count),
    }


def weighted_bce_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """가중치 적용 BCE Loss"""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    weighted = bce * weights
    return weighted.mean()


def finetune_sa_head(
    base_model: str = None,
    output_model: str = None,
    input_dir: str = None,
    freeze_pa: bool = True,
    freeze_pd: bool = True,
    freeze_encoder: bool = False,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    trace_path: str = None,
):
    """SA Head Fine-tuning 메인 함수"""
    
    # 기본 경로 설정
    if base_model is None:
        base_model = MODELS_ROOT / "boundary_multitask.pt"
    else:
        base_model = Path(base_model)
    
    if output_model is None:
        output_model = MODELS_ROOT / "boundary_sa_finetuned.pt"
    else:
        output_model = Path(output_model)
    
    if input_dir is None:
        input_dir = DATASETS_ROOT / "sa_boundary_hardneg_v2"
    else:
        input_dir = Path(input_dir)
    
    print(f"📂 Input: {input_dir}")
    print(f"🔧 Base model: {base_model}")
    print(f"💾 Output model: {output_model}")
    print(f"⚙️ Freeze PA: {freeze_pa}, Freeze PD: {freeze_pd}, Freeze Encoder: {freeze_encoder}")
    print(f"⚙️ Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
    
    # 디바이스 설정
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # 기존 모델 로드
    ckpt = torch.load(base_model, map_location=device, weights_only=False)
    vocab = ckpt["vocab"]
    max_len = ckpt.get("max_len", 1024)
    tasks = ckpt.get("tasks", ["pa", "sa", "pd"])
    
    print(f"✅ Base model loaded (vocab_size={len(vocab)}, max_len={max_len})")
    
    # 모델 생성 및 가중치 로드
    model = MultiHeadBoundary(
        vocab_size=len(vocab) + 1,
        hidden_dim=128,
        emb_dim=64,
        tasks=tasks
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    
    # PA/PD head 동결
    frozen_params = 0
    trainable_params = 0
    
    if freeze_pa and "pa" in model.heads:
        for param in model.heads["pa"].parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        print("🔒 PA head frozen")
    
    if freeze_pd and "pd" in model.heads:
        for param in model.heads["pd"].parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        print("🔒 PD head frozen")
    
    if freeze_encoder:
        for param in model.emb.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        for param in model.lstm.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        print("🔒 Encoder frozen")
    
    # 학습 가능 파라미터 수 계산
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"📊 Frozen params: {frozen_params:,}")
    print(f"📊 Trainable params: {trainable_params:,}")
    
    # 데이터 로드
    train_samples = load_jsonl(input_dir / "train.jsonl", task="sa")
    val_samples = load_jsonl(input_dir / "val.jsonl", task="sa")
    
    print(f"📊 Train samples: {len(train_samples)}")
    print(f"📊 Val samples: {len(val_samples)}")
    
    train_ds = SAFinetuneDataset(train_samples, vocab, max_len)
    val_ds = SAFinetuneDataset(val_samples, vocab, max_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Optimizer (학습 가능 파라미터만)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # 학습 기록
    history = []
    best_f1 = 0.0
    best_state = None
    
    # 학습 루프
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        
        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            
            optimizer.zero_grad()
            logits = model(x, "sa")
            loss = weighted_bce_loss(logits, y, w)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
        
        avg_loss = total_loss / max(1, steps)
        
        # 검증
        val_scores = evaluate(model, val_loader, device, task="sa")
        
        print(f"Epoch {epoch}/{epochs}: loss={avg_loss:.4f} | SA: P={val_scores['p']:.4f} R={val_scores['r']:.4f} F1={val_scores['f']:.4f}")
        
        # 기록
        history.append({
            "epoch": epoch,
            "loss": float(avg_loss),
            "val_sa": val_scores,
        })
        
        # Best 모델 저장
        if val_scores["f"] > best_f1:
            best_f1 = val_scores["f"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ⭐ New best: F1={best_f1:.4f}")
    
    # PA 성능 보존 검증 (동결했더라도 확인)
    print("\n🔍 PA 성능 보존 검증...")
    if (input_dir.parent / "pa_boundary" / "val.jsonl").exists():
        try:
            pa_samples = load_jsonl(input_dir.parent / "pa_boundary" / "val.jsonl", task="pa")
            pa_ds = SAFinetuneDataset(pa_samples, vocab, max_len)
            pa_loader = DataLoader(pa_ds, batch_size=batch_size, shuffle=False)
            
            model.eval()
            pa_scores = evaluate(model, pa_loader, device, task="pa")
            print(f"   PA val F1: {pa_scores['f']:.4f} (should be ~0.97)")
            
            # 원래 PA 성능과 비교
            original_pa_f1 = ckpt.get("test_scores", {}).get("pa", {}).get("f", 0.97)
            if pa_scores["f"] < original_pa_f1 - 0.01:
                print(f"   ⚠️ WARNING: PA 성능 저하 감지! ({original_pa_f1:.4f} → {pa_scores['f']:.4f})")
            else:
                print(f"   ✅ PA 성능 보존 확인")
        except Exception as e:
            print(f"   ⚠️ PA 검증 스킵: {e}")
    
    # 최종 모델 저장
    if best_state is None:
        best_state = model.state_dict()
    
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "vocab": vocab,
        "max_len": max_len,
        "tasks": tasks,
        "history": history,
        "best_sa_f1": best_f1,
        "freeze_config": {
            "freeze_pa": freeze_pa,
            "freeze_pd": freeze_pd,
            "freeze_encoder": freeze_encoder,
        },
        "training_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "input_dir": str(input_dir),
        },
        "timestamp": datetime.now().isoformat(),
    }, output_model)
    
    print(f"\n💾 Model saved: {output_model}")
    print(f"🏆 Best SA F1: {best_f1:.4f}")
    
    # Trace 저장
    if trace_path:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as f:
            for h in history:
                h["timestamp"] = datetime.now().isoformat()
                h["stage"] = "finetune_sa"
                f.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(f"📝 Trace saved: {trace_path}")
    
    print("\n✅ SA Head Fine-tuning 완료!")
    
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SA Head Fine-tuning (PA/PD Head 동결)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="기존 모델 경로 (기본: models/boundary_multitask.pt)")
    parser.add_argument("--output-model", type=str, default=None,
                        help="출력 모델 경로 (기본: models/boundary_sa_finetuned.pt)")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Hard Negative 데이터 디렉토리 (기본: datasets/sa_boundary_hardneg_v2)")
    parser.add_argument("--no-freeze-pa", action="store_true",
                        help="PA head 동결 해제")
    parser.add_argument("--no-freeze-pd", action="store_true",
                        help="PD head 동결 해제")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Encoder(emb+lstm) 동결")
    parser.add_argument("--epochs", type=int, default=5,
                        help="학습 epoch 수 (기본: 5)")
    parser.add_argument("--batch", type=int, default=32,
                        help="배치 크기 (기본: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="학습률 (기본: 1e-4)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="디바이스 (기본: cuda)")
    parser.add_argument("--trace", type=str, default=None,
                        help="Trace JSONL 출력 경로")
    
    args = parser.parse_args()
    
    finetune_sa_head(
        base_model=args.base_model,
        output_model=args.output_model,
        input_dir=args.input_dir,
        freeze_pa=not args.no_freeze_pa,
        freeze_pd=not args.no_freeze_pd,
        freeze_encoder=args.freeze_encoder,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        trace_path=args.trace,
    )
