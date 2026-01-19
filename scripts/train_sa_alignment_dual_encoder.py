#!/usr/bin/env python3
"""Train dual-encoder alignment model for SA using TRAIN-ONLY CSV.

핵심 원칙:
- 학습 데이터는 train만 사용 (val/test 사용 금지)
- 원문(src)과 번역문(tgt) 쌍을 동시에 학습
- Hard Negative: 같은 배치 내 다른 쌍을 negative로 사용
- 출력: models/dual_encoder_alignment_sa.pt

PA 버전(train_alignment_dual_encoder_trainonly.py)을 SA에 맞게 수정.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = WORKSPACE_ROOT / "models"


class SaTrainCsvDataset(Dataset):
    """SA 학습용 데이터셋 (원문-번역문 구 단위 쌍)"""
    
    def __init__(
        self,
        csv_path: Path,
        build_vocab: bool = False,
        vocab_src=None,
        vocab_tgt=None,
        max_len: int = 256,  # SA는 구 단위라 PA보다 짧음
        enable_hard_neg: bool = False,
        hard_neg_mode: str = "prefix_token",
    ):
        self.csv_path = csv_path
        self.max_len = max_len
        self.enable_hard_neg = enable_hard_neg
        self.hard_neg_mode = hard_neg_mode
        
        # CSV 로드
        df = pd.read_csv(csv_path, dtype=str)
        df = df.fillna("")
        
        # SA 컬럼: 원문, 번역문
        self.src_texts = df["원문"].tolist()
        self.tgt_texts = df["번역문"].tolist()
        
        # 빈 문자열 제거
        valid_pairs = [(s, t) for s, t in zip(self.src_texts, self.tgt_texts) if s.strip() and t.strip()]
        self.src_texts = [p[0] for p in valid_pairs]
        self.tgt_texts = [p[1] for p in valid_pairs]
        
        print(f"📂 Loaded {len(self.src_texts)} SA pairs from {csv_path}")
        
        # Vocab 구축 또는 재사용
        if build_vocab:
            src_chars = set()
            tgt_chars = set()
            for s in self.src_texts:
                src_chars.update(list(s))
            for t in self.tgt_texts:
                tgt_chars.update(list(t))
            self.vocab_src = {c: i + 1 for i, c in enumerate(sorted(src_chars))}
            self.vocab_tgt = {c: i + 1 for i, c in enumerate(sorted(tgt_chars))}
        else:
            self.vocab_src = vocab_src or {}
            self.vocab_tgt = vocab_tgt or {}
        
        # Hard Negative 가중치 (기본 1.0)
        self.HN = [torch.tensor(1.0, dtype=torch.float32) for _ in self.src_texts]
    
    def encode_text(self, t: str, vocab) -> torch.Tensor:
        ids = [vocab.get(ch, 0) for ch in t][:self.max_len]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_enc = self.encode_text(self.src_texts[idx], self.vocab_src)
        tgt_enc = self.encode_text(self.tgt_texts[idx], self.vocab_tgt)
        return src_enc, tgt_enc, self.HN[idx]


class CharEncoder(nn.Module):
    """문자 단위 인코더 (BiLSTM)"""
    
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden * 2, 256)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        out, _ = self.lstm(e)
        # Mean pooling over sequence
        mask = (x != 0).float().unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.proj(pooled)


class DualEncoder(nn.Module):
    """Dual Encoder: 원문/번역문 각각 인코딩 후 유사도 계산"""
    
    def __init__(self, vocab_src: int, vocab_tgt: int):
        super().__init__()
        self.enc_src = CharEncoder(vocab_src)
        self.enc_tgt = CharEncoder(vocab_tgt)
    
    def forward(self, xs: torch.Tensor, xt: torch.Tensor):
        v_src = self.enc_src(xs)  # [B, 256]
        v_tgt = self.enc_tgt(xt)  # [B, 256]
        return v_src, v_tgt


def main():
    parser = argparse.ArgumentParser(description="Train SA Dual-Encoder Alignment")
    parser.add_argument("--train-csv", type=str, default="datasets/s2p/train.csv",
                        help="학습용 SA CSV 경로")
    parser.add_argument("--epochs", type=int, default=10, help="에포크 수")
    parser.add_argument("--batch", type=int, default=128, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument("--max-len", type=int, default=256, help="최대 시퀀스 길이")
    parser.add_argument("--device", type=str, default="cuda", help="디바이스")
    parser.add_argument("--out", type=str, default="models/dual_encoder_alignment_sa.pt",
                        help="출력 모델 경로")
    parser.add_argument("--trace", type=str, default=None, help="Trace JSONL 경로")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # 데이터 로드
    train_csv = WORKSPACE_ROOT / args.train_csv
    train_ds = SaTrainCsvDataset(train_csv, build_vocab=True, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    
    print(f"📊 Vocab sizes: src={len(train_ds.vocab_src)}, tgt={len(train_ds.vocab_tgt)}")
    
    # 모델 초기화
    model = DualEncoder(len(train_ds.vocab_src), len(train_ds.vocab_tgt)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Trace 기록용
    trace_records = []
    
    # 학습 루프
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for xs, xt, _ in train_loader:
            xs = xs.to(device)
            xt = xt.to(device)
            
            v_src, v_tgt = model(xs, xt)
            
            # Contrastive loss: In-batch negatives
            # 정답: 대각선 (i-th src와 i-th tgt가 매칭)
            # 오답: 같은 배치 내 다른 쌍
            sim = torch.matmul(v_src, v_tgt.T)  # [B, B]
            labels = torch.arange(xs.size(0), device=device)
            
            loss_src = nn.CrossEntropyLoss()(sim, labels)
            loss_tgt = nn.CrossEntropyLoss()(sim.T, labels)
            loss = (loss_src + loss_tgt) / 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f}")
        
        # Trace 기록
        if args.trace:
            from datetime import datetime
            trace_records.append({
                "epoch": epoch,
                "loss": avg_loss,
                "timestamp": datetime.now().isoformat(),
                "stage": "train_sa_alignment",
            })
    
    # 모델 저장
    out_path = WORKSPACE_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "vocab_src": train_ds.vocab_src,
        "vocab_tgt": train_ds.vocab_tgt,
        "max_len": args.max_len,
        "config": {
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
        },
    }, out_path)
    print(f"💾 Model saved: {out_path}")
    
    # Trace 저장
    if args.trace:
        import json
        trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as f:
            for r in trace_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"📝 Trace saved: {trace_path}")
    
    print("✅ SA Dual-Encoder Alignment 학습 완료!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
