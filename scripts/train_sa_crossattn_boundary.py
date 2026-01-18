#!/usr/bin/env python3
"""SA Cross-Attention 경계 모델 학습

원문-번역문 쌍을 입력으로 받아 번역문의 구 경계를 예측
- 원문과 번역문 간 Cross-Attention으로 의미 대응 학습
- 원문 구 구조를 참조하여 번역문 경계 결정

Usage:
    python scripts/train_sa_crossattn_boundary.py --epochs 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"


def load_sa_phrase_pairs(csv_path: Path) -> List[Dict]:
    """
    SA CSV를 로드하여 문장별로:
    - 전체 원문 (구들 연결)
    - 전체 번역문 (구들 연결)
    - 번역문 B/O 레이블
    
    Returns:
        [{"src": "원문", "tgt": "번역문", "labels": "BOOOOB..."}, ...]
    """
    df = pd.read_csv(csv_path)
    
    # 문장별로 그룹핑
    sent_groups = defaultdict(list)
    for _, row in df.iterrows():
        sent_id = row['문장식별자']
        phrase_id = row['구식별자']
        src = str(row['원문']).strip()
        tgt = str(row['번역문']).strip()
        sent_groups[sent_id].append((phrase_id, src, tgt))
    
    samples = []
    for sent_id, phrases in sent_groups.items():
        # 구식별자 순으로 정렬
        phrases.sort(key=lambda x: x[0])
        
        full_src = ""
        full_tgt = ""
        labels = ""
        
        for i, (_, src_phrase, tgt_phrase) in enumerate(phrases):
            if not src_phrase and not tgt_phrase:
                continue
            
            # 구 사이 공백 추가
            if i > 0:
                if full_src:
                    full_src += " "
                if full_tgt:
                    full_tgt += " "
                    labels += "O"  # 공백은 O
            
            full_src += src_phrase
            
            # 번역문: 첫 문자 B, 나머지 O
            for j, char in enumerate(tgt_phrase):
                full_tgt += char
                labels += "B" if j == 0 else "O"
        
        if full_src and full_tgt and len(full_tgt) == len(labels):
            samples.append({
                "src": full_src,
                "tgt": full_tgt,
                "labels": labels,
                "num_phrases": len(phrases),
            })
    
    return samples


def build_vocab(samples: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """원문/번역문 각각 vocab 구축"""
    src_chars = set()
    tgt_chars = set()
    for s in samples:
        src_chars.update(list(s["src"]))
        tgt_chars.update(list(s["tgt"]))
    
    src_vocab = {c: i + 1 for i, c in enumerate(sorted(src_chars))}
    tgt_vocab = {c: i + 1 for i, c in enumerate(sorted(tgt_chars))}
    return src_vocab, tgt_vocab


class CrossAttnBoundaryDataset(Dataset):
    def __init__(self, samples: List[Dict], src_vocab: Dict, tgt_vocab: Dict, 
                 src_max_len: int, tgt_max_len: int):
        self.samples = samples
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
    
    def __len__(self):
        return len(self.samples)
    
    def _encode(self, text: str, vocab: Dict, max_len: int) -> torch.Tensor:
        ids = [vocab.get(ch, 0) for ch in text][:max_len]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)
    
    def _encode_labels(self, labels: str, max_len: int) -> torch.Tensor:
        arr = [1.0 if ch == "B" else 0.0 for ch in labels][:max_len]
        arr += [0.0] * (max_len - len(arr))
        return torch.tensor(arr, dtype=torch.float)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        src = self._encode(s["src"], self.src_vocab, self.src_max_len)
        tgt = self._encode(s["tgt"], self.tgt_vocab, self.tgt_max_len)
        labels = self._encode_labels(s["labels"], self.tgt_max_len)
        tgt_len = min(len(s["tgt"]), self.tgt_max_len)
        return src, tgt, labels, tgt_len


class CrossAttnBoundaryModel(nn.Module):
    """Cross-Attention 기반 경계 태거
    
    1. Source Encoder: 원문 인코딩
    2. Target Encoder: 번역문 인코딩
    3. Cross-Attention: 원문 참조
    4. Boundary Head: 경계 예측
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 emb_dim: int = 128, hidden: int = 256, num_heads: int = 4):
        super().__init__()
        
        # Embeddings
        self.src_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=0)
        
        # Source encoder (BiLSTM)
        self.src_encoder = nn.LSTM(
            emb_dim, hidden // 2, num_layers=2, 
            bidirectional=True, batch_first=True, dropout=0.2
        )
        
        # Target encoder (BiLSTM)
        self.tgt_encoder = nn.LSTM(
            emb_dim, hidden // 2, num_layers=2, 
            bidirectional=True, batch_first=True, dropout=0.2
        )
        
        # Cross-Attention: target attends to source
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden)
        
        # Boundary classifier
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),  # concat(tgt_hidden, cross_attn_out)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) 원문 토큰 IDs
            tgt: (batch, tgt_len) 번역문 토큰 IDs
        
        Returns:
            logits: (batch, tgt_len) 각 번역문 문자의 경계 logit
        """
        # Encode source
        src_emb = self.src_emb(src)  # (B, S, E)
        src_hidden, _ = self.src_encoder(src_emb)  # (B, S, H)
        
        # Encode target
        tgt_emb = self.tgt_emb(tgt)  # (B, T, E)
        tgt_hidden, _ = self.tgt_encoder(tgt_emb)  # (B, T, H)
        
        # Cross-attention: target queries, source keys/values
        # Create key_padding_mask for source (True = padding)
        src_padding_mask = (src == 0)  # (B, S)
        
        cross_out, _ = self.cross_attn(
            query=tgt_hidden,
            key=src_hidden,
            value=src_hidden,
            key_padding_mask=src_padding_mask
        )  # (B, T, H)
        
        cross_out = self.norm(cross_out + tgt_hidden)  # residual + norm
        
        # Concatenate target hidden and cross-attention output
        combined = torch.cat([tgt_hidden, cross_out], dim=-1)  # (B, T, H*2)
        
        # Predict boundary
        logits = self.boundary_head(combined).squeeze(-1)  # (B, T)
        
        return logits


def compute_f1(logits: torch.Tensor, labels: torch.Tensor, 
               lengths: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float]:
    """마스킹된 위치 제외하고 F1 계산"""
    preds = torch.sigmoid(logits) >= threshold
    
    tp = fp = fn = 0
    batch_size = logits.shape[0]
    
    for i in range(batch_size):
        length = lengths[i].item()
        pred = preds[i, :length]
        gold = labels[i, :length] > 0.5
        
        tp += (pred & gold).sum().item()
        fp += (pred & ~gold).sum().item()
        fn += (~pred & gold).sum().item()
    
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def evaluate(model, loader, device):
    model.eval()
    total_p, total_r, total_f = 0, 0, 0
    n = 0
    with torch.no_grad():
        for src, tgt, labels, lengths in loader:
            src, tgt, labels = src.to(device), tgt.to(device), labels.to(device)
            logits = model(src, tgt)
            p, r, f = compute_f1(logits, labels, lengths)
            total_p += p
            total_r += r
            total_f += f
            n += 1
    return {"p": total_p / n, "r": total_r / n, "f1": total_f / n}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SA Cross-Attention 경계 모델 학습")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--src-max-len", type=int, default=256)
    parser.add_argument("--tgt-max-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # 데이터 로드
    print("📂 Loading SA phrase pair data...")
    train_samples = load_sa_phrase_pairs(DATASETS_ROOT / "sa" / "train.csv")
    val_samples = load_sa_phrase_pairs(DATASETS_ROOT / "sa" / "val.csv")
    test_samples = load_sa_phrase_pairs(DATASETS_ROOT / "sa" / "test.csv")
    
    print(f"📊 Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # 샘플 확인
    if train_samples:
        s = train_samples[0]
        print(f"   예시:")
        print(f"     src='{s['src'][:50]}...'")
        print(f"     tgt='{s['tgt'][:50]}...'")
        print(f"     labels='{s['labels'][:50]}...'")
        print(f"     num_phrases={s['num_phrases']}")
    
    # Vocab
    src_vocab, tgt_vocab = build_vocab(train_samples + val_samples)
    print(f"📚 Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}")
    
    # 데이터셋
    train_ds = CrossAttnBoundaryDataset(train_samples, src_vocab, tgt_vocab, 
                                         args.src_max_len, args.tgt_max_len)
    val_ds = CrossAttnBoundaryDataset(val_samples, src_vocab, tgt_vocab,
                                       args.src_max_len, args.tgt_max_len)
    test_ds = CrossAttnBoundaryDataset(test_samples, src_vocab, tgt_vocab,
                                        args.src_max_len, args.tgt_max_len)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    
    # 모델
    model = CrossAttnBoundaryModel(
        src_vocab_size=len(src_vocab) + 1,
        tgt_vocab_size=len(tgt_vocab) + 1,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model params: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Positive weight for class imbalance (B is rare)
    pos_weight = torch.tensor([5.0]).to(device)  # B 레이블 가중치
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_f1 = 0
    best_state = None
    
    print(f"\n🚀 학습 시작 (epochs={args.epochs})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for src, tgt, labels, lengths in train_loader:
            src, tgt, labels = src.to(device), tgt.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(src, tgt)
            
            # 마스킹: 패딩 위치 제외
            mask = torch.arange(logits.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1).to(device)
            
            loss = criterion(logits[mask], labels[mask])
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        val_scores = evaluate(model, val_loader, device)
        print(f"  [Epoch {epoch:2d}] loss={total_loss/len(train_loader):.4f} | val P={val_scores['p']:.4f} R={val_scores['r']:.4f} F1={val_scores['f1']:.4f}")
        
        if val_scores['f1'] > best_f1:
            best_f1 = val_scores['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # 최고 모델로 테스트
    if best_state:
        model.load_state_dict(best_state)
    test_scores = evaluate(model, test_loader, device)
    print(f"\n📈 Test: P={test_scores['p']:.4f} R={test_scores['r']:.4f} F1={test_scores['f1']:.4f}")
    
    # 저장
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_ROOT / "sa_crossattn_boundary.pt"
    torch.save({
        "state_dict": best_state if best_state else model.state_dict(),
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "src_max_len": args.src_max_len,
        "tgt_max_len": args.tgt_max_len,
        "hidden": args.hidden,
        "emb_dim": args.emb_dim,
        "test_scores": test_scores,
    }, save_path)
    print(f"💾 Saved: {save_path}")


if __name__ == "__main__":
    main()
