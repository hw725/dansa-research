#!/usr/bin/env python3
"""SA PLM 기반 Cross-Attention 경계 모델 학습

원문: SikuBERT (han문 전용)
번역문: KLUE-RoBERTa (한국어 한자 처리)

Usage:
    python scripts/train_sa_plm_boundary.py --epochs 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"


class PLMCrossAttnBoundaryModel(nn.Module):
    """PLM 기반 Cross-Attention 경계 태거
    
    원문: SikuBERT (SIKU-BERT/sikubert)
    번역문: KLUE-RoBERTa (klue/roberta-large)
    """
    
    def __init__(
        self, 
        src_model_name: str = "SIKU-BERT/sikubert",
        tgt_model_name: str = "klue/roberta-large",
        hidden_dim: int = 256,
        num_heads: int = 4,
        freeze_plm: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.freeze_plm = freeze_plm
        
        # PLM 로드
        print(f"🔧 원문 PLM 로드 중: {src_model_name}")
        self.src_tokenizer = AutoTokenizer.from_pretrained(src_model_name, trust_remote_code=True)
        self.src_plm = AutoModel.from_pretrained(src_model_name, trust_remote_code=True)
        src_hidden = self.src_plm.config.hidden_size
        
        print(f"🔧 번역문 PLM 로드 중: {tgt_model_name}")
        self.tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)
        self.tgt_plm = AutoModel.from_pretrained(tgt_model_name)
        tgt_hidden = self.tgt_plm.config.hidden_size
        
        # PLM Freeze (선택적)
        if freeze_plm:
            for param in self.src_plm.parameters():
                param.requires_grad = False
            for param in self.tgt_plm.parameters():
                param.requires_grad = False
            print("❄️ PLM 파라미터 Freeze")
        
        # 차원 맞춤 프로젝션
        self.src_proj = nn.Linear(src_hidden, hidden_dim)
        self.tgt_proj = nn.Linear(tgt_hidden, hidden_dim)
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 경계 예측 Head
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        print(f"✅ 모델 초기화 완료 (src_hidden={src_hidden}, tgt_hidden={tgt_hidden}, proj={hidden_dim})")
    
    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask):
        """
        Args:
            src_input_ids: (B, S) 원문 토큰 ID
            src_attention_mask: (B, S) 원문 어텐션 마스크
            tgt_input_ids: (B, T) 번역문 토큰 ID
            tgt_attention_mask: (B, T) 번역문 어텐션 마스크
            
        Returns:
            logits: (B, T) 각 토큰의 경계 확률 (logits)
        """
        # PLM 인코딩
        with torch.set_grad_enabled(not self.freeze_plm):
            src_outputs = self.src_plm(input_ids=src_input_ids, attention_mask=src_attention_mask)
            src_hidden = src_outputs.last_hidden_state  # (B, S, H_src)
            
            tgt_outputs = self.tgt_plm(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask)
            tgt_hidden = tgt_outputs.last_hidden_state  # (B, T, H_tgt)
        
        # 차원 프로젝션
        src_proj = self.src_proj(src_hidden)  # (B, S, D)
        tgt_proj = self.tgt_proj(tgt_hidden)  # (B, T, D)
        
        # Cross-Attention (번역문 -> 원문)
        # key_padding_mask: True인 위치는 무시됨
        key_padding_mask = (src_attention_mask == 0)
        cross_out, _ = self.cross_attn(
            query=tgt_proj,
            key=src_proj,
            value=src_proj,
            key_padding_mask=key_padding_mask
        )
        
        # Residual + Norm
        cross_out = self.norm(cross_out + tgt_proj)
        
        # Boundary Head
        combined = torch.cat([tgt_proj, cross_out], dim=-1)  # (B, T, 2D)
        logits = self.boundary_head(combined).squeeze(-1)  # (B, T)
        
        return logits


class SABoundaryDataset(Dataset):
    """SA 경계 데이터셋"""
    
    def __init__(
        self, 
        samples: List[Dict], 
        src_tokenizer, 
        tgt_tokenizer,
        src_max_len: int = 128,
        tgt_max_len: int = 256
    ):
        self.samples = samples
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 원문 토크나이징
        src_enc = self.src_tokenizer(
            sample['src'],
            max_length=self.src_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 번역문 토크나이징
        tgt_enc = self.tgt_tokenizer(
            sample['tgt'],
            max_length=self.tgt_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 레이블 생성: 토큰별 경계 레이블
        # 문자 레이블 -> 토큰 레이블 매핑 필요
        char_labels = sample['labels']  # "BOOOOBOO..."
        
        # 토큰-문자 매핑
        token_labels = self._map_char_labels_to_tokens(
            sample['tgt'], 
            char_labels
        )
        
        # 패딩
        if len(token_labels) < self.tgt_max_len:
            token_labels = token_labels + [0] * (self.tgt_max_len - len(token_labels))
        else:
            token_labels = token_labels[:self.tgt_max_len]
        
        return {
            'src_input_ids': src_enc['input_ids'].squeeze(0),
            'src_attention_mask': src_enc['attention_mask'].squeeze(0),
            'tgt_input_ids': tgt_enc['input_ids'].squeeze(0),
            'tgt_attention_mask': tgt_enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(token_labels, dtype=torch.float),
        }
    
    def _map_char_labels_to_tokens(self, text: str, char_labels: str) -> List[int]:
        """문자 레이블을 토큰 레이블로 매핑"""
        # 토큰화 (offset 포함)
        enc = self.tgt_tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            max_length=self.tgt_max_len,
            truncation=True
        )
        
        offsets = enc.get('offset_mapping', [])
        token_labels = []
        
        for offset in offsets:
            if offset is None or offset == (0, 0):
                # 특수 토큰
                token_labels.append(0)
            else:
                start, end = offset
                if start < len(char_labels):
                    # 토큰 시작 위치의 레이블 사용
                    label = 1 if char_labels[start] == 'B' else 0
                    token_labels.append(label)
                else:
                    token_labels.append(0)
        
        return token_labels


def load_sa_phrase_pairs(csv_path: Path) -> List[Dict]:
    """SA CSV 로드하여 문장별 샘플 생성"""
    df = pd.read_csv(csv_path)
    
    sent_groups = defaultdict(list)
    for _, row in df.iterrows():
        sent_id = row['문장식별자']
        phrase_id = row['구식별자']
        src = str(row['원문']).strip()
        tgt = str(row['번역문']).strip()
        sent_groups[sent_id].append((phrase_id, src, tgt))
    
    samples = []
    for sent_id, phrases in sent_groups.items():
        phrases.sort(key=lambda x: x[0])
        
        full_src = ""
        full_tgt = ""
        labels = ""
        
        for i, (_, src_phrase, tgt_phrase) in enumerate(phrases):
            if not src_phrase and not tgt_phrase:
                continue
            
            if i > 0:
                full_src += " "
                full_tgt += " "
                labels += "O"  # 공백은 경계 아님
            
            full_src += src_phrase
            
            # 번역문 레이블: 첫 문자가 B, 나머지 O
            if tgt_phrase:
                labels += "B" + "O" * (len(tgt_phrase) - 1)
                full_tgt += tgt_phrase
        
        if full_src and full_tgt and labels:
            samples.append({
                'src': full_src,
                'tgt': full_tgt,
                'labels': labels,
                'sent_id': sent_id
            })
    
    return samples


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # 디바이스 이동
        src_input_ids = batch['src_input_ids'].to(device)
        src_attention_mask = batch['src_attention_mask'].to(device)
        tgt_input_ids = batch['tgt_input_ids'].to(device)
        tgt_attention_mask = batch['tgt_attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask)
        
        # Loss (valid 토큰만)
        mask = tgt_attention_mask.bool()
        loss = criterion(logits[mask], labels[mask])
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device) -> Tuple[float, float]:
    """평가"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src_input_ids = batch['src_input_ids'].to(device)
            src_attention_mask = batch['src_attention_mask'].to(device)
            tgt_input_ids = batch['tgt_input_ids'].to(device)
            tgt_attention_mask = batch['tgt_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask)
            
            mask = tgt_attention_mask.bool()
            loss = criterion(logits[mask], labels[mask])
            total_loss += loss.item()
            
            # Accuracy
            preds = (torch.sigmoid(logits[mask]) > 0.5).float()
            correct += (preds == labels[mask]).sum().item()
            total += mask.sum().item()
    
    return total_loss / len(dataloader), correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--freeze-plm", action="store_true", default=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    train_samples = load_sa_phrase_pairs(DATASETS_ROOT / "sa" / "train.csv")
    val_samples = load_sa_phrase_pairs(DATASETS_ROOT / "sa" / "val.csv")
    test_samples = load_sa_phrase_pairs(DATASETS_ROOT / "sa" / "test.csv")
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # 모델 생성
    print("\n🏗️ 모델 생성 중...")
    model = PLMCrossAttnBoundaryModel(
        src_model_name="SIKU-BERT/sikubert",
        tgt_model_name="klue/roberta-large",
        hidden_dim=args.hidden_dim,
        freeze_plm=args.freeze_plm,
        device=device
    ).to(device)
    
    # 데이터셋
    train_dataset = SABoundaryDataset(
        train_samples, model.src_tokenizer, model.tgt_tokenizer
    )
    val_dataset = SABoundaryDataset(
        val_samples, model.src_tokenizer, model.tgt_tokenizer
    )
    test_dataset = SABoundaryDataset(
        test_samples, model.src_tokenizer, model.tgt_tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 학습 설정
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    # 학습
    print("\n🚀 학습 시작")
    print("=" * 60)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📌 Epoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Best 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            save_path = MODELS_ROOT / "sa_plm_boundary.pt"
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'src_model_name': "SIKU-BERT/sikubert",
                'tgt_model_name': "klue/roberta-large",
                'hidden_dim': args.hidden_dim,
            }, save_path)
            print(f"  💾 Best 모델 저장: {save_path}")
    
    # 테스트
    print("\n" + "=" * 60)
    print("📊 테스트 평가")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Best Epoch: {best_epoch}")
    
    print("\n✅ 학습 완료!")


if __name__ == "__main__":
    main()
