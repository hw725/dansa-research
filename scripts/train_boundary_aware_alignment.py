#!/usr/bin/env python3
"""
Context-aware Alignment Model (Boundary 정보 포함)

기존 DualEncoder와의 차이점:
1. 입력: (src_tokens, src_boundaries) + (tgt_tokens, tgt_boundaries)
2. Boundary embedding 추가 (학습 가능한 positional encoding)
3. Loss: contrastive loss + boundary match loss (multitask)

학습 목표:
- 의미 유사도 (contrastive learning)
- 경계 일치 여부 (binary classification)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
MODELS_ROOT = WORKSPACE_ROOT / "models"


class BoundaryAwareDataset(Dataset):
    """Boundary 정보를 포함한 Alignment 데이터셋"""
    
    def __init__(
        self,
        jsonl_path: Path,
        build_vocab: bool = False,
        vocab_src: Dict[str, int] = None,
        vocab_tgt: Dict[str, int] = None,
        max_len: int = 512,
    ):
        if not jsonl_path.exists():
            raise FileNotFoundError(f"파일 없음: {jsonl_path}")
        
        self.max_len = max_len
        self.samples = []
        
        # JSONL 로드
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        print(f"✅ 로드: {len(self.samples):,}개 샘플")
        
        # Vocab 구축 (학습 시)
        if build_vocab:
            chars_src = set()
            chars_tgt = set()
            for sample in tqdm(self.samples, desc="Building vocab"):
                chars_src.update(list(sample.get("src", "")))
                chars_tgt.update(list(sample.get("tgt", "")))
            
            self.vocab_src = {c: i + 1 for i, c in enumerate(sorted(chars_src))}
            self.vocab_tgt = {c: i + 1 for i, c in enumerate(sorted(chars_tgt))}
            print(f"   vocab_src: {len(self.vocab_src):,}자")
            print(f"   vocab_tgt: {len(self.vocab_tgt):,}자")
        else:
            self.vocab_src = vocab_src
            self.vocab_tgt = vocab_tgt
        
        # 인코딩
        self._encode_all()
    
    def _encode_all(self):
        """모든 샘플을 텐서로 인코딩"""
        self.X_src = []
        self.X_tgt = []
        self.B_src = []  # boundary flags
        self.B_tgt = []
        self.labels = []
        self.boundary_matches = []
        
        for sample in tqdm(self.samples, desc="Encoding"):
            src = sample.get("src", "")
            tgt = sample.get("tgt", "")
            src_boundaries = sample.get("src_boundaries", [])
            tgt_boundaries = sample.get("tgt_boundaries", [])
            label = sample.get("label", 1)
            boundary_match = sample.get("boundary_match", 1)
            
            # Text encoding
            src_ids = self._encode_text(src, self.vocab_src)
            tgt_ids = self._encode_text(tgt, self.vocab_tgt)
            
            # Boundary flags (1 if position is boundary, 0 otherwise)
            src_flags = self._encode_boundaries(len(src), src_boundaries)
            tgt_flags = self._encode_boundaries(len(tgt), tgt_boundaries)
            
            self.X_src.append(src_ids)
            self.X_tgt.append(tgt_ids)
            self.B_src.append(src_flags)
            self.B_tgt.append(tgt_flags)
            self.labels.append(torch.tensor(label, dtype=torch.long))
            self.boundary_matches.append(torch.tensor(boundary_match, dtype=torch.float32))
    
    def _encode_text(self, text: str, vocab: Dict[str, int]) -> torch.Tensor:
        """텍스트를 character ID로 변환"""
        ids = [vocab.get(ch, 0) for ch in text]
        ids = ids[:self.max_len]
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
        return torch.tensor(ids, dtype=torch.long)
    
    def _encode_boundaries(self, text_len: int, boundaries: List[int]) -> torch.Tensor:
        """Boundary 위치를 binary flag로 변환"""
        flags = [0] * min(text_len, self.max_len)
        for b in boundaries:
            if 0 <= b < len(flags):
                flags[b] = 1
        
        # Padding
        pad_len = self.max_len - len(flags)
        if pad_len > 0:
            flags += [0] * pad_len
        
        return torch.tensor(flags, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X_src)
    
    def __getitem__(self, idx):
        return (
            self.X_src[idx], 
            self.X_tgt[idx],
            self.B_src[idx],
            self.B_tgt[idx],
            self.labels[idx],
            self.boundary_matches[idx]
        )


class BoundaryAwareCharEncoder(nn.Module):
    """
    Boundary 정보를 포함한 Character Encoder
    
    입력:
    - x: character IDs [B, L]
    - b: boundary flags [B, L]
    
    출력:
    - z: normalized embedding [B, D]
    """
    
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Boundary embedding (학습 가능한 binary embedding)
        self.boundary_emb = nn.Embedding(2, emb_dim // 2)  # [0=no boundary, 1=boundary]
        
        # LSTM
        self.lstm = nn.LSTM(
            emb_dim + emb_dim // 2,  # char + boundary
            hidden, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Projection
        self.proj = nn.Linear(hidden * 2, 256)
    
    def forward(self, x, b):
        """
        Args:
            x: [B, L] character IDs
            b: [B, L] boundary flags (0 or 1)
        Returns:
            z: [B, 256] normalized embedding
        """
        # Character embedding
        char_emb = self.char_emb(x)  # [B, L, emb_dim]
        
        # Boundary embedding
        b_int = b.long()  # Convert to int for embedding
        bound_emb = self.boundary_emb(b_int)  # [B, L, emb_dim//2]
        
        # Concatenate
        combined = torch.cat([char_emb, bound_emb], dim=-1)  # [B, L, emb_dim + emb_dim//2]
        
        # LSTM
        lstm_out, _ = self.lstm(combined)  # [B, L, hidden*2]
        
        # Mean pooling
        pooled = lstm_out.mean(dim=1)  # [B, hidden*2]
        
        # Projection & normalization
        z = self.proj(pooled)  # [B, 256]
        z = nn.functional.normalize(z, dim=-1)
        
        return z


class BoundaryAwareDualEncoder(nn.Module):
    """
    Context-aware Dual Encoder
    
    두 가지 출력:
    1. Similarity score (contrastive learning)
    2. Boundary match score (binary classification)
    """
    
    def __init__(self, vocab_src: int, vocab_tgt: int):
        super().__init__()
        self.enc_src = BoundaryAwareCharEncoder(vocab_src)
        self.enc_tgt = BoundaryAwareCharEncoder(vocab_tgt)
        
        # Boundary match classifier
        self.boundary_classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, xs, xt, bs, bt, compute_boundary_match=True):
        """
        Args:
            xs: [B, L] src character IDs
            xt: [B, L] tgt character IDs
            bs: [B, L] src boundary flags
            bt: [B, L] tgt boundary flags
            compute_boundary_match: boundary classifier 실행 여부
        
        Returns:
            zs: [B, 256] src embedding
            zt: [B, 256] tgt embedding
            boundary_score: [B] boundary match probability (optional)
        """
        zs = self.enc_src(xs, bs)
        zt = self.enc_tgt(xt, bt)
        
        if compute_boundary_match:
            # Concatenate embeddings for boundary classifier
            combined = torch.cat([zs, zt], dim=-1)  # [B, 512]
            boundary_score = self.boundary_classifier(combined).squeeze(-1)  # [B]
            return zs, zt, boundary_score
        else:
            return zs, zt


def train_one_epoch(
    model: BoundaryAwareDualEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 0.07,
    boundary_loss_weight: float = 0.3,
    max_steps: int = 0
) -> Dict[str, float]:
    """1 epoch 학습"""
    model.train()
    
    total_loss = 0.0
    total_contrast = 0.0
    total_boundary = 0.0
    steps_ran = 0
    
    for step_i, (xs, xt, bs, bt, labels, boundary_matches) in enumerate(loader, 1):
        steps_ran = step_i
        
        xs = xs.to(device)
        xt = xt.to(device)
        bs = bs.to(device)
        bt = bt.to(device)
        boundary_matches = boundary_matches.to(device)
        
        # Forward
        zs, zt, boundary_scores = model(xs, xt, bs, bt, compute_boundary_match=True)
        
        # 1. Contrastive loss (in-batch negatives)
        sim_mat = zs @ zt.t()  # [B, B]
        logits = sim_mat / temperature
        target = torch.arange(logits.size(0), device=device)
        contrast_loss = nn.functional.cross_entropy(logits, target)
        
        # 2. Boundary match loss (binary cross-entropy)
        boundary_loss = nn.functional.binary_cross_entropy(
            boundary_scores, 
            boundary_matches
        )
        
        # Combined loss
        loss = contrast_loss + boundary_loss_weight * boundary_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_contrast += contrast_loss.item()
        total_boundary += boundary_loss.item()
        
        if max_steps > 0 and step_i >= max_steps:
            break
    
    denom = max(1, steps_ran)
    return {
        "loss": total_loss / denom,
        "contrast": total_contrast / denom,
        "boundary": total_boundary / denom
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train boundary-aware alignment model")
    parser.add_argument(
        "--train-jsonl",
        default=str(DATASETS_ROOT / "alignment" / "pa" / "train_boundary_aware.jsonl"),
        help="학습 데이터 경로"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--boundary-weight", type=float, default=0.3, 
                       help="Boundary loss weight (0~1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default=str(MODELS_ROOT / "dual_encoder_boundary_aware_pa.pt"),
        help="출력 모델 경로"
    )
    
    args = parser.parse_args()
    
    # Seed
    seed = args.seed
    if seed != 0:
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except:
            pass
    
    train_jsonl = Path(args.train_jsonl)
    if not train_jsonl.exists():
        raise FileNotFoundError(f"학습 데이터 없음: {train_jsonl}")
    
    # 안전장치: test/val 파일을 학습 데이터로 넣는 사고 방지
    lower = str(train_jsonl).replace("\\", "/").lower()
    if "/test" in lower or "/val" in lower or lower.endswith("test.jsonl") or lower.endswith("val.jsonl"):
        raise ValueError(
            f"❌ 학습에는 train 파일만 사용할 수 있습니다.\n"
            f"입력 파일: {train_jsonl}\n"
            f"train_boundary_aware.jsonl 파일을 지정하세요."
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Dataset
    train_ds = BoundaryAwareDataset(
        train_jsonl,
        build_vocab=True,
        max_len=512
    )
    
    # DataLoader
    loader_generator = None
    if seed != 0:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        generator=loader_generator,
        num_workers=0
    )
    
    # Model
    model = BoundaryAwareDualEncoder(
        vocab_src=len(train_ds.vocab_src) + 1,
        vocab_tgt=len(train_ds.vocab_tgt) + 1
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print()
    print(f"📊 학습 설정:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch}")
    print(f"   - Max steps: {args.max_steps if args.max_steps > 0 else 'unlimited'}")
    print(f"   - Temperature: {args.temperature}")
    print(f"   - Boundary loss weight: {args.boundary_weight}")
    print()
    
    # Training
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            device,
            temperature=args.temperature,
            boundary_loss_weight=args.boundary_weight,
            max_steps=args.max_steps
        )
        
        print(f"Epoch {epoch:2d}: "
              f"loss={metrics['loss']:.4f} "
              f"(contrast={metrics['contrast']:.4f}, "
              f"boundary={metrics['boundary']:.4f})")
    
    # Save
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_src": train_ds.vocab_src,
            "vocab_tgt": train_ds.vocab_tgt,
        },
        output_path
    )
    
    print()
    print(f"✅ 저장 완료: {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
