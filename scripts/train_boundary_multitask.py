#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
멀티태스크 경계 태깅 학습 스크립트
- 공유 인코더 + 태스크별(head) 로짓
- 태스크: pa(문단→문장), sa(문장→구), pd(문단 경계, 가중치 낮음)
- 입력은 *_boundary/train|val|test.jsonl (tgt 기준)
- 출력 체크포인트: models/boundary_multitask.pt (state_dict, vocab, max_len)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"

TASKS = {
    "pa": "pa_boundary",
    "sa": "sa_boundary",
    "pd": "pd_boundary",
}


class BoundarySample:
    def __init__(self, text: str, labels: str, task: str):
        self.text = text
        self.labels = labels
        self.task = task


def load_jsonl(path: Path, task: str) -> List[BoundarySample]:
    samples: List[BoundarySample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(BoundarySample(obj["text"], obj["labels"], task))
    return samples


def build_vocab(train_sets: Dict[str, List[BoundarySample]]) -> Dict[str, int]:
    chars = set()
    for samples in train_sets.values():
        for s in samples:
            chars.update(list(s.text))
    return {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0: pad


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    ids = [vocab.get(ch, 0) for ch in text][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def encode_labels(labels: str, max_len: int) -> torch.Tensor:
    arr = [1 if ch == "B" else 0 for ch in labels][:max_len]
    if len(arr) < max_len:
        arr += [0] * (max_len - len(arr))
    return torch.tensor(arr, dtype=torch.float)


class MultiTaskDataset(Dataset):
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
        return x, y, s.task


class MultiHeadBoundary(nn.Module):
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
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    tp = (preds * labels).sum().item()
    fp = (preds * (1 - labels)).sum().item()
    fn = ((1 - preds) * labels).sum().item()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def evaluate(model: MultiHeadBoundary, loader: DataLoader, device: torch.device) -> Dict[str, Dict[str, float]]:
    model.eval()
    totals: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    with torch.no_grad():
        for x, y, tasks in loader:
            x = x.to(device)
            y = y.to(device)
            for t in set(tasks):
                mask = [i for i, tt in enumerate(tasks) if tt == t]
                x_t = x[mask]
                y_t = y[mask]
                logits = model(x_t, t)
                p, r, f = f1_from_probs(logits, y_t)
                if t not in totals:
                    totals[t] = {"p": 0.0, "r": 0.0, "f": 0.0}
                    counts[t] = 0
                totals[t]["p"] += p
                totals[t]["r"] += r
                totals[t]["f"] += f
                counts[t] += 1
    for t in totals:
        totals[t] = {k: v / counts[t] for k, v in totals[t].items()}
    return totals


def collate_fn(batch):
    xs, ys, ts = zip(*batch)
    x = torch.stack(xs)
    y = torch.stack(ys)
    return x, y, list(ts)


def train_multitask(epochs: int = 3, batch_size: int = 32, max_len: int = 1024, pd_weight: float = 0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 데이터 로드
    splits = {"train": {}, "val": {}, "test": {}}
    for name, ds_dir in TASKS.items():
        for split in ["train", "val", "test"]:
            path = DATASETS_ROOT / ds_dir / f"{split}.jsonl"
            if not path.exists():
                splits[split][name] = []
                continue
            splits[split][name] = load_jsonl(path, name)

    train_sets = {k: v for k, v in splits["train"].items()}
    vocab = build_vocab(train_sets)
    print(f"📚 Vocab size: {len(vocab)}")

    # 2) 데이터셋/로더 생성
    loaders = {}
    for split in ["train", "val", "test"]:
        merged: List[BoundarySample] = []
        for t, samples in splits[split].items():
            merged.extend(samples)
        ds = MultiTaskDataset(merged, vocab, max_len=max_len)
        shuffle = split == "train"
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    # 3) 모델
    model = MultiHeadBoundary(vocab_size=len(vocab) + 1, hidden_dim=128, emb_dim=64, tasks=list(TASKS.keys())).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    task_weights = {"pa": 1.0, "sa": 1.0, "pd": pd_weight}

    history = []

    # 4) 학습 루프
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for x, y, tasks in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            # 배치 내 여러 태스크를 분리 처리
            loss = 0.0
            for t in set(tasks):
                mask = [i for i, tt in enumerate(tasks) if tt == t]
                x_t = x[mask]
                y_t = y[mask]
                logits = model(x_t, t)
                l = bce(logits, y_t) * task_weights.get(t, 1.0)
                loss += l
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            steps += 1
        avg_loss = total_loss / max(1, steps)

        val_scores = evaluate(model, loaders["val"], device)
        log_parts = [f"{t}:F1={val_scores.get(t, {}).get('f', 0):.4f}" for t in TASKS.keys()]
        print(f"Epoch {ep}: loss={avg_loss:.4f} | " + " | ".join(log_parts))

        history.append({
            "epoch": ep,
            "loss": float(avg_loss),
            "val": val_scores,
        })

    # 5) 테스트 평가
    test_scores = evaluate(model, loaders["test"], device)
    print("Test scores:")
    for t in TASKS.keys():
        sc = test_scores.get(t, {"p": 0, "r": 0, "f": 0})
        print(f"  {t}: P={sc['p']:.4f} R={sc['r']:.4f} F1={sc['f']:.4f}")

    # 6) 저장
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_ROOT / "boundary_multitask.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "max_len": max_len,
        "task_weights": task_weights,
        "tasks": list(TASKS.keys()),
        "test_scores": test_scores,
        "history": history,
    }, out_path)
    print(f"💾 Saved: {out_path}")

    # 메트릭 저장
    metrics_path = MODELS_ROOT / "boundary_multitask_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(test_scores, f, indent=2, ensure_ascii=False)
    print(f"📝 Metrics: {metrics_path}")

    history_path = MODELS_ROOT / "boundary_multitask_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"📝 History: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="멀티태스크 경계 태깅 학습")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--pd-weight", type=float, default=0.3)
    args = parser.parse_args()

    train_multitask(epochs=args.epochs, batch_size=args.batch, max_len=args.max_len, pd_weight=args.pd_weight)
