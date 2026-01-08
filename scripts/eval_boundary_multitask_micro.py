#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Boundary multitask 모델을 padding 제외 micro P/R/F1로 평가.

배경:
- scripts/train_boundary_multitask.py의 evaluate()는 배치 단위 평균으로 집계되어
  데이터셋 전체 micro 관점과 다를 수 있음.
- 또한 패딩 구간을 포함하면 모델이 과대평가될 수 있어, 실제 label 길이(=원문 길이)
  기준으로 마스킹하여 점수를 산출한다.

사용 예:
  python scripts/eval_boundary_multitask_micro.py --split test
  python scripts/eval_boundary_multitask_micro.py --task pa --split val --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = REPO_ROOT / "datasets"
MODELS_ROOT = REPO_ROOT / "models"


@dataclass
class Sample:
    text: str
    labels: str


def load_jsonl(path: Path) -> List[Sample]:
    out: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(Sample(text=obj["text"], labels=obj["labels"]))
    return out


def _encode_text(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    ids = [vocab.get(ch, 0) for ch in text][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def _encode_labels(labels: str, max_len: int) -> torch.Tensor:
    arr = [1 if ch == "B" else 0 for ch in labels][:max_len]
    if len(arr) < max_len:
        arr += [0] * (max_len - len(arr))
    return torch.tensor(arr, dtype=torch.float)


class BoundaryEvalDataset(Dataset):
    def __init__(self, samples: List[Sample], vocab: Dict[str, int], max_len: int):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        effective_len = min(len(s.text), len(s.labels), self.max_len)
        x = _encode_text(s.text, self.vocab, self.max_len)
        y = _encode_labels(s.labels, self.max_len)
        return x, y, effective_len


def collate_fn(batch):
    xs, ys, lens = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.tensor(lens, dtype=torch.long)


def micro_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def evaluate_micro(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    threshold: float,
    ignore_pos0: bool,
) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    total_tokens = 0
    total_pos = 0
    with torch.no_grad():
        for x, y, lens in loader:
            x = x.to(device)
            y = y.to(device)
            lens = lens.to(device)

            logits = model(x, task)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            max_len = y.size(1)
            pos_idx = torch.arange(max_len, device=device).unsqueeze(0).expand(y.size(0), -1)
            mask = pos_idx < lens.unsqueeze(1)
            if ignore_pos0:
                mask[:, 0] = False

            y_bin = (y >= 0.5)
            preds_bin = (preds >= 0.5)

            tp += int(((preds_bin & y_bin) & mask).sum().item())
            fp += int(((preds_bin & (~y_bin)) & mask).sum().item())
            fn += int((((~preds_bin) & y_bin) & mask).sum().item())

            total_tokens += int(mask.sum().item())
            total_pos += int((y_bin & mask).sum().item())

    p, r, f1 = micro_prf(tp, fp, fn)
    return {
        "p": float(p),
        "r": float(r),
        "f": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tokens": int(total_tokens),
        "pos": int(total_pos),
        "pos_rate": float(total_pos / (total_tokens if total_tokens else 1)),
    }


def maybe_overlap(train_path: Path, split_path: Path) -> Dict[str, int]:
    if not train_path.exists() or not split_path.exists():
        return {}
    train = load_jsonl(train_path)
    split = load_jsonl(split_path)
    train_texts = {s.text for s in train}
    split_texts = {s.text for s in split}
    return {
        "train": len(train_texts),
        "split": len(split_texts),
        "overlap": len(train_texts & split_texts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate boundary_multitask with padding-masked micro metrics")
    parser.add_argument("--checkpoint", type=str, default=str(MODELS_ROOT / "boundary_multitask.pt"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--task", type=str, default="all", help="pa|sa|pd|all")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ignore-pos0", action="store_true", default=True)
    parser.add_argument("--include-pos0", action="store_true", help="pos0도 평가에 포함")
    parser.add_argument("--check-overlap", action="store_true", help="train과의 텍스트 중복을 간단 체크")
    args = parser.parse_args()

    ignore_pos0 = args.ignore_pos0 and (not args.include_pos0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab: Dict[str, int] = ckpt["vocab"]
    max_len: int = int(ckpt.get("max_len", 1024))
    tasks: List[str] = list(ckpt.get("tasks", ["pa", "sa", "pd"]))

    # 추론 모델 구조는 common/boundary_model_loader.py와 동일하게 맞춘다.
    from common.boundary_model_loader import MultiHeadBoundary  # 로컬 import (repo 루트 기준)

    model = MultiHeadBoundary(vocab_size=len(vocab) + 1, tasks=tasks).to(device)

    state_dict = ckpt.get("state_dict", ckpt)
    needs_remap = any(k.startswith("emb.") or k.startswith("lstm.") for k in state_dict.keys())
    if needs_remap:
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("emb.") or k.startswith("lstm."):
                remapped[f"encoder.{k}"] = v
            else:
                remapped[k] = v
        state_dict = remapped
        print("🔧 Checkpoint 키 재매핑 수행 (emb./lstm. → encoder.*)")

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    task_list = tasks if args.task == "all" else [args.task]
    for t in task_list:
        if t not in tasks:
            raise ValueError(f"Unknown task '{t}'. Available: {tasks}")

    results = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "threshold": float(args.threshold),
        "max_len": int(max_len),
        "ignore_pos0": bool(ignore_pos0),
        "tasks": {},
    }

    for task in task_list:
        ds_dir = DATASETS_ROOT / f"{task}_boundary"
        split_path = ds_dir / f"{args.split}.jsonl"
        if not split_path.exists():
            print(f"⚠️  skip: {split_path} (not found)")
            continue

        samples = load_jsonl(split_path)
        ds = BoundaryEvalDataset(samples, vocab=vocab, max_len=max_len)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

        metrics = evaluate_micro(model, loader, device=device, task=task, threshold=args.threshold, ignore_pos0=ignore_pos0)
        results["tasks"][task] = metrics

        print(
            f"[{task}/{args.split}] micro P={metrics['p']:.4f} R={metrics['r']:.4f} F1={metrics['f']:.4f} "
            f"(tokens={metrics['tokens']}, pos_rate={metrics['pos_rate']:.4f})"
        )

        if args.check_overlap and args.split != "train":
            ov = maybe_overlap(ds_dir / "train.jsonl", split_path)
            if ov:
                print(f"  overlap(train vs {args.split}): {ov['overlap']}/{ov['split']} (train_unique={ov['train']})")

    print("\nJSON summary:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
