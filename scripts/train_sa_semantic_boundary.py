#!/usr/bin/env python3
"""SA 의미 대응 기반 경계 모델 학습

BGE 임베딩을 활용하여 원문 구와 번역문 구의 의미적 대응을 학습
- 원문: 구 단위 BGE 임베딩 시퀀스
- 번역문: 문자 레벨 인코딩
- 출력: 번역문의 각 문자가 새로운 구의 시작인지 예측

Usage:
    python scripts/train_sa_semantic_boundary.py --epochs 20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import pickle

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"


def compute_phrase_embeddings(csv_path: Path, cache_path: Path = None):
    """각 구의 BGE 임베딩 계산"""
    if cache_path and cache_path.exists():
        print(f"📂 Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    from common.embedders.bge import get_embed_func
    embed_func = get_embed_func()
    
    df = pd.read_csv(csv_path)
    
    # 모든 구 텍스트 수집
    all_src = [str(row['원문']).strip() for _, row in df.iterrows()]
    all_tgt = [str(row['번역문']).strip() for _, row in df.iterrows()]
    
    print(f"  Computing embeddings for {len(all_src)} phrases...")
    
    # 배치 임베딩 계산
    src_embs = embed_func(all_src, batch_size=64)
    tgt_embs = embed_func(all_tgt, batch_size=64)
    
    # 문장별로 그룹핑
    sent_groups = defaultdict(list)
    for i, (_, row) in enumerate(df.iterrows()):
        sent_id = row['문장식별자']
        phrase_id = row['구식별자']
        sent_groups[sent_id].append({
            'phrase_id': phrase_id,
            'src_text': all_src[i],
            'tgt_text': all_tgt[i],
            'src_emb': np.array(src_embs[i]),
            'tgt_emb': np.array(tgt_embs[i]),
        })
    
    # 문장별 샘플 생성
    samples = []
    for sent_id, phrases in sent_groups.items():
        phrases.sort(key=lambda x: x['phrase_id'])
        
        # 원문 구 임베딩 시퀀스
        src_phrase_embs = [p['src_emb'] for p in phrases]
        
        # 번역문 연결 및 레이블
        full_tgt = ""
        labels = ""
        phrase_boundaries = []  # 각 구의 시작 위치
        
        for i, p in enumerate(phrases):
            tgt_phrase = p['tgt_text']
            if not tgt_phrase:
                continue
            
            # 구 사이 공백 추가
            if i > 0 and full_tgt:
                full_tgt += " "
                labels += "O"
            
            phrase_boundaries.append(len(full_tgt))
            
            for j, char in enumerate(tgt_phrase):
                full_tgt += char
                labels += "B" if j == 0 else "O"
        
        if full_tgt and len(full_tgt) == len(labels):
            samples.append({
                'sent_id': sent_id,
                'src_phrase_embs': np.stack(src_phrase_embs),  # (N, emb_dim)
                'tgt_text': full_tgt,
                'labels': labels,
                'phrase_boundaries': phrase_boundaries,
                'num_phrases': len(phrases),
            })
    
    if cache_path:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"  Cached to {cache_path}")
    
    return samples


class SemanticBoundaryDataset(Dataset):
    def __init__(self, samples: List[Dict], tgt_vocab: Dict, 
                 max_phrases: int, tgt_max_len: int, emb_dim: int):
        self.samples = samples
        self.tgt_vocab = tgt_vocab
        self.max_phrases = max_phrases
        self.tgt_max_len = tgt_max_len
        self.emb_dim = emb_dim
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 원문 구 임베딩 (N, emb_dim) -> (max_phrases, emb_dim)
        src_embs = s['src_phrase_embs']
        n_phrases = min(src_embs.shape[0], self.max_phrases)
        src_padded = np.zeros((self.max_phrases, self.emb_dim), dtype=np.float32)
        src_padded[:n_phrases] = src_embs[:n_phrases]
        
        # 번역문 인코딩
        tgt_ids = [self.tgt_vocab.get(ch, 0) for ch in s['tgt_text']][:self.tgt_max_len]
        tgt_ids += [0] * (self.tgt_max_len - len(tgt_ids))
        
        # 레이블
        labels = [1.0 if ch == "B" else 0.0 for ch in s['labels']][:self.tgt_max_len]
        labels += [0.0] * (self.tgt_max_len - len(labels))
        
        tgt_len = min(len(s['tgt_text']), self.tgt_max_len)
        
        return (
            torch.tensor(src_padded, dtype=torch.float32),
            torch.tensor(tgt_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(n_phrases, dtype=torch.long),
            torch.tensor(tgt_len, dtype=torch.long),
        )


class SemanticBoundaryModel(nn.Module):
    """의미 대응 기반 경계 모델
    
    원문: 구 단위 BGE 임베딩 시퀀스
    번역문: 문자 레벨 인코딩
    Cross-Attention으로 의미 대응 학습
    """
    def __init__(self, tgt_vocab_size: int, emb_dim: int = 128, 
                 hidden: int = 256, src_emb_dim: int = 1024, num_heads: int = 4):
        super().__init__()
        
        # 원문 구 임베딩 projection
        self.src_proj = nn.Linear(src_emb_dim, hidden)
        
        # 번역문 문자 임베딩
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=0)
        
        # 번역문 인코더
        self.tgt_encoder = nn.LSTM(
            emb_dim, hidden // 2, num_layers=2,
            bidirectional=True, batch_first=True, dropout=0.2
        )
        
        # Cross-Attention: 번역문이 원문 구를 참조
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        
        self.norm = nn.LayerNorm(hidden)
        
        # 경계 예측
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, src_embs: torch.Tensor, tgt_ids: torch.Tensor, 
                n_phrases: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_embs: (batch, max_phrases, src_emb_dim) 원문 구 임베딩
            tgt_ids: (batch, tgt_len) 번역문 문자 IDs
            n_phrases: (batch,) 각 샘플의 실제 구 개수
        """
        batch_size = src_embs.shape[0]
        max_phrases = src_embs.shape[1]
        
        # 원문 구 임베딩 projection
        src_hidden = self.src_proj(src_embs)  # (B, N, H)
        
        # 번역문 인코딩
        tgt_emb = self.tgt_emb(tgt_ids)  # (B, T, E)
        tgt_hidden, _ = self.tgt_encoder(tgt_emb)  # (B, T, H)
        
        # 원문 패딩 마스크
        phrase_mask = torch.arange(max_phrases, device=src_embs.device).unsqueeze(0) >= n_phrases.unsqueeze(1)
        
        # Cross-Attention
        cross_out, attn_weights = self.cross_attn(
            query=tgt_hidden,
            key=src_hidden,
            value=src_hidden,
            key_padding_mask=phrase_mask
        )
        
        cross_out = self.norm(cross_out + tgt_hidden)
        combined = torch.cat([tgt_hidden, cross_out], dim=-1)
        
        logits = self.boundary_head(combined).squeeze(-1)
        
        return logits, attn_weights


def compute_f1(logits, labels, lengths, threshold=0.5):
    preds = torch.sigmoid(logits) >= threshold
    tp = fp = fn = 0
    for i in range(logits.shape[0]):
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-phrases", type=int, default=50)
    parser.add_argument("--tgt-max-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # 임베딩 캐시 경로
    cache_dir = CACHE_ROOT / "sa_semantic"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("📂 Loading data with BGE embeddings...")
    train_samples = compute_phrase_embeddings(
        DATASETS_ROOT / "sa" / "train.csv",
        cache_dir / "train_embs.pkl"
    )
    val_samples = compute_phrase_embeddings(
        DATASETS_ROOT / "sa" / "val.csv",
        cache_dir / "val_embs.pkl"
    )
    test_samples = compute_phrase_embeddings(
        DATASETS_ROOT / "sa" / "test.csv",
        cache_dir / "test_embs.pkl"
    )
    
    print(f"📊 Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Vocab
    tgt_chars = set()
    for s in train_samples + val_samples:
        tgt_chars.update(list(s['tgt_text']))
    tgt_vocab = {c: i + 1 for i, c in enumerate(sorted(tgt_chars))}
    print(f"📚 Target vocab: {len(tgt_vocab)}")
    
    # 임베딩 차원 확인
    emb_dim = train_samples[0]['src_phrase_embs'].shape[1]
    print(f"🔢 BGE embedding dim: {emb_dim}")
    
    # 데이터셋
    train_ds = SemanticBoundaryDataset(train_samples, tgt_vocab, args.max_phrases, args.tgt_max_len, emb_dim)
    val_ds = SemanticBoundaryDataset(val_samples, tgt_vocab, args.max_phrases, args.tgt_max_len, emb_dim)
    test_ds = SemanticBoundaryDataset(test_samples, tgt_vocab, args.max_phrases, args.tgt_max_len, emb_dim)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    
    # 모델
    model = SemanticBoundaryModel(
        tgt_vocab_size=len(tgt_vocab) + 1,
        src_emb_dim=emb_dim,
        hidden=256,
    ).to(device)
    
    print(f"🧠 Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    pos_weight = torch.tensor([5.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_f1 = 0
    best_state = None
    
    print(f"\n🚀 학습 시작 (epochs={args.epochs})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for src_embs, tgt_ids, labels, n_phrases, tgt_lens in train_loader:
            src_embs = src_embs.to(device)
            tgt_ids = tgt_ids.to(device)
            labels = labels.to(device)
            n_phrases = n_phrases.to(device)
            tgt_lens = tgt_lens.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(src_embs, tgt_ids, n_phrases)
            
            mask = torch.arange(logits.shape[1], device=device).unsqueeze(0) < tgt_lens.unsqueeze(1)
            loss = criterion(logits[mask], labels[mask])
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_p, val_r, val_f1 = 0, 0, 0
        n = 0
        with torch.no_grad():
            for src_embs, tgt_ids, labels, n_phrases, tgt_lens in val_loader:
                src_embs = src_embs.to(device)
                tgt_ids = tgt_ids.to(device)
                labels = labels.to(device)
                n_phrases = n_phrases.to(device)
                
                logits, _ = model(src_embs, tgt_ids, n_phrases)
                p, r, f = compute_f1(logits, labels, tgt_lens)
                val_p += p
                val_r += r
                val_f1 += f
                n += 1
        
        val_f1 /= n
        print(f"  [Epoch {epoch:2d}] loss={total_loss/len(train_loader):.4f} | val F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_p, test_r, test_f1 = 0, 0, 0
    n = 0
    with torch.no_grad():
        for src_embs, tgt_ids, labels, n_phrases, tgt_lens in test_loader:
            src_embs = src_embs.to(device)
            tgt_ids = tgt_ids.to(device)
            labels = labels.to(device)
            n_phrases = n_phrases.to(device)
            
            logits, _ = model(src_embs, tgt_ids, n_phrases)
            p, r, f = compute_f1(logits, labels, tgt_lens)
            test_p += p
            test_r += r
            test_f1 += f
            n += 1
    
    test_f1 /= n
    test_p /= n
    test_r /= n
    print(f"\n📈 Test: P={test_p:.4f} R={test_r:.4f} F1={test_f1:.4f}")
    
    # 저장
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_ROOT / "sa_semantic_boundary.pt"
    torch.save({
        "state_dict": best_state if best_state else model.state_dict(),
        "tgt_vocab": tgt_vocab,
        "max_phrases": args.max_phrases,
        "tgt_max_len": args.tgt_max_len,
        "src_emb_dim": emb_dim,
        "test_scores": {"p": test_p, "r": test_r, "f1": test_f1},
    }, save_path)
    print(f"💾 Saved: {save_path}")


if __name__ == "__main__":
    main()
