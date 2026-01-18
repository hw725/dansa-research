#!/usr/bin/env python3
"""SA 구 경계 모델 학습 (올바른 버전)

datasets/sa/train.csv (구 단위 Gold)에서 번역문 경계 학습
- 같은 문장식별자 내의 여러 구를 연결하여 경계 생성
- 각 구의 시작점 = 경계(B)

Usage:
    python scripts/train_sa_phrase_boundary.py --epochs 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"


def load_sa_csv_as_phrase_data(csv_path: Path, text_col: str = "번역문") -> List[Dict]:
    """
    SA CSV를 로드하여 문장별로 그룹핑, B/O 레이블 생성
    
    🆕 공백 보존: 구 사이에 공백 추가 (실제 입력과 동일하게)
    
    Returns:
        [{"text": "전체 문장 번역문", "labels": "BOOOOBOOOB..."}, ...]
    """
    df = pd.read_csv(csv_path)
    
    # 문장별로 그룹핑
    sent_groups = defaultdict(list)
    for _, row in df.iterrows():
        sent_id = row['문장식별자']
        phrase_id = row['구식별자']
        text = str(row[text_col])
        sent_groups[sent_id].append((phrase_id, text))
    
    samples = []
    for sent_id, phrases in sent_groups.items():
        # 구식별자 순으로 정렬
        phrases.sort(key=lambda x: x[0])
        
        # 전체 텍스트와 레이블 생성
        full_text = ""
        labels = ""
        
        for i, (_, phrase_text) in enumerate(phrases):
            phrase_text = phrase_text.strip()
            if not phrase_text:
                continue
            
            # 🆕 구 사이에 공백 추가 (첫 구 제외)
            if i > 0 and full_text:
                full_text += " "
                labels += "O"  # 공백은 경계가 아님
            
            for j, char in enumerate(phrase_text):
                full_text += char
                # 구의 첫 문자 = 경계(B), 나머지 = O
                if j == 0:
                    labels += "B"
                else:
                    labels += "O"
        
        if full_text and len(full_text) == len(labels):
            samples.append({"text": full_text, "labels": labels})
    
    return samples


def build_vocab(samples: List[Dict]) -> Dict[str, int]:
    chars = set()
    for s in samples:
        chars.update(list(s["text"]))
    return {c: i + 1 for i, c in enumerate(sorted(chars))}


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    ids = [vocab.get(ch, 0) for ch in text][:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def encode_labels(labels: str, max_len: int) -> torch.Tensor:
    arr = [1.0 if ch == "B" else 0.0 for ch in labels][:max_len]
    arr += [0.0] * (max_len - len(arr))
    return torch.tensor(arr, dtype=torch.float)


class PhraseBoundaryDataset(Dataset):
    def __init__(self, samples: List[Dict], vocab: Dict[str, int], max_len: int):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return encode_text(s["text"], self.vocab, self.max_len), encode_labels(s["labels"], self.max_len)


class PhraseBoundaryTagger(nn.Module):
    """BiLSTM 기반 구 경계 태거"""
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.emb(x))
        return self.fc(h).squeeze(-1)


def compute_f1(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float]:
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * labels).sum().item()
    fp = (preds * (1 - labels)).sum().item()
    fn = ((1 - preds) * labels).sum().item()
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def evaluate(model, loader, device):
    model.eval()
    total_p, total_r, total_f = 0, 0, 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            p, r, f = compute_f1(logits, y)
            total_p += p
            total_r += r
            total_f += f
            n += 1
    return {"p": total_p / n, "r": total_r / n, "f1": total_f / n}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SA 구 경계 모델 학습")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    # SA CSV에서 구 경계 데이터 로드
    data_dir = DATASETS_ROOT / "sa"
    
    print("📂 Loading SA gold data...")
    train_samples = load_sa_csv_as_phrase_data(data_dir / "train.csv")
    val_samples = load_sa_csv_as_phrase_data(data_dir / "val.csv")
    test_samples = load_sa_csv_as_phrase_data(data_dir / "test.csv")
    
    print(f"📊 Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # 샘플 확인
    if train_samples:
        s = train_samples[1]
        print(f"   예시: text='{s['text'][:50]}...' labels='{s['labels'][:50]}...'")

    # Vocab
    vocab = build_vocab(train_samples)
    print(f"📚 Vocab: {len(vocab)}")

    # 데이터셋
    train_ds = PhraseBoundaryDataset(train_samples, vocab, args.max_len)
    val_ds = PhraseBoundaryDataset(val_samples, vocab, args.max_len)
    test_ds = PhraseBoundaryDataset(test_samples, vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)

    # 모델
    model = PhraseBoundaryTagger(len(vocab) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    best_state = None

    print(f"\n🚀 학습 시작 (epochs={args.epochs})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_scores = evaluate(model, val_loader, device)
        print(f"  [Epoch {epoch}] loss={total_loss/len(train_loader):.4f} | val F1={val_scores['f1']:.4f}")

        if val_scores['f1'] > best_f1:
            best_f1 = val_scores['f1']
            best_state = model.state_dict().copy()

    # 최고 모델로 테스트
    if best_state:
        model.load_state_dict(best_state)
    test_scores = evaluate(model, test_loader, device)
    print(f"\n📈 Test: P={test_scores['p']:.4f} R={test_scores['r']:.4f} F1={test_scores['f1']:.4f}")

    # 저장 (기존 모델 덮어쓰기)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_ROOT / "sa_boundary_tagger.pt"
    torch.save({
        "state_dict": best_state if best_state else model.state_dict(),
        "vocab": vocab,
        "max_len": args.max_len,
        "test_scores": test_scores,
    }, save_path)
    print(f"💾 Saved: {save_path}")


if __name__ == "__main__":
    main()
