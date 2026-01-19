#!/usr/bin/env python3
"""Train dual-encoder alignment model using TRAIN-ONLY CSV and auto-generated negatives.

핵심 원칙:
- 학습 데이터는 train만 사용 (val/test 사용 금지)
- negative 라벨이 따로 없어도 in-batch negatives(contrastive loss)로 자동 생성

기본 입력:
- datasets/p2s/train.csv (columns: 원문, 번역문, book_name, ...)

출력:
- models/dual_encoder_alignment_pa.pt (pa.processor strict 모드에서 사용)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = WORKSPACE_ROOT / "models"


class PaTrainCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        build_vocab: bool = False,
        vocab_src=None,
        vocab_tgt=None,
        max_len: int = 512,
        enable_hard_neg: bool = False,
        hard_neg_mode: str = "prefix_token",
    ):
        import pandas as pd

        if not csv_path.exists():
            raise FileNotFoundError(f"train.csv 파일이 없습니다: {csv_path}")

        df = pd.read_csv(csv_path, dtype=str).fillna("")
        required = {"원문", "번역문"}
        if not required.issubset(set(df.columns)):
            missing = sorted(required - set(df.columns))
            raise ValueError(f"train.csv에 필수 컬럼이 없습니다: {missing} (필요: 원문, 번역문)")

        self.max_len = int(max_len)
        self.enable_hard_neg = bool(enable_hard_neg)
        self.hard_neg_mode = str(hard_neg_mode)

        self.src_texts: List[str] = [str(x) for x in df["원문"].tolist()]
        self.tgt_texts: List[str] = [str(x) for x in df["번역문"].tolist()]

        # 하드 네거티브(경계 쉬프트)용: 인접 샘플에서 next src prefix를 붙인다.
        # - same book_name + same 문단식별자(가능하면) 조건으로 더 안전하게 생성
        self._hard_neg_src_texts: List[Optional[str]] = [None for _ in self.src_texts]
        if self.enable_hard_neg:
            has_book = "book_name" in df.columns
            has_pid = "문단식별자" in df.columns
            books = [str(x) for x in df["book_name"].tolist()] if has_book else ["" for _ in self.src_texts]
            pids = [str(x) for x in df["문단식별자"].tolist()] if has_pid else ["" for _ in self.src_texts]
            for i in range(len(self.src_texts) - 1):
                if has_book and books[i] != books[i + 1]:
                    continue
                if has_pid and pids[i] != pids[i + 1]:
                    continue
                src_a = self.src_texts[i]
                src_b = self.src_texts[i + 1]
                if not src_a.strip() or not src_b.strip():
                    continue
                prefix = self._take_prefix(src_b, mode=self.hard_neg_mode)
                if not prefix.strip():
                    continue
                self._hard_neg_src_texts[i] = (src_a.rstrip() + " " + prefix.lstrip()).strip()

        # vocab
        if build_vocab:
            chars_src = set()
            chars_tgt = set()
            for t in self.src_texts:
                chars_src.update(list(t))
            if self.enable_hard_neg:
                for t in self._hard_neg_src_texts:
                    if t:
                        chars_src.update(list(t))
            for t in self.tgt_texts:
                chars_tgt.update(list(t))
            self.vocab_src = {c: i + 1 for i, c in enumerate(sorted(chars_src))}
            self.vocab_tgt = {c: i + 1 for i, c in enumerate(sorted(chars_tgt))}
        else:
            self.vocab_src = vocab_src
            self.vocab_tgt = vocab_tgt

        self.X_src = [self.encode_text(t, self.vocab_src) for t in self.src_texts]
        self.X_tgt = [self.encode_text(t, self.vocab_tgt) for t in self.tgt_texts]

        if self.enable_hard_neg:
            self.X_src_hardneg = [
                self.encode_text(t, self.vocab_src) if t else torch.zeros((self.max_len,), dtype=torch.long)
                for t in self._hard_neg_src_texts
            ]
            self.HN = [torch.tensor(1.0 if t else 0.0, dtype=torch.float32) for t in self._hard_neg_src_texts]
        else:
            self.X_src_hardneg = [torch.zeros((self.max_len,), dtype=torch.long) for _ in self.src_texts]
            self.HN = [torch.tensor(0.0, dtype=torch.float32) for _ in self.src_texts]

    @staticmethod
    def _take_prefix(text: str, mode: str = "prefix_token") -> str:
        t = (text or "").strip()
        if not t:
            return ""
        if mode == "full":
            return t
        parts = t.split()
        return parts[0] if parts else t

    def encode_text(self, t: str, vocab):
        ids = [vocab.get(ch, 0) for ch in t]
        ids = ids[: self.max_len]
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [0] * pad_len
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.X_src)

    def __getitem__(self, idx):
        # (xs, xt, xs_hn, hn)
        return self.X_src[idx], self.X_tgt[idx], self.X_src_hardneg[idx], self.HN[idx]


class CharEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden * 2, 256)

    def forward(self, x):
        e = self.emb(x)
        o, _ = self.lstm(e)
        m = o.mean(dim=1)
        z = self.proj(m)
        z = nn.functional.normalize(z, dim=-1)
        return z


class DualEncoder(nn.Module):
    def __init__(self, vocab_src: int, vocab_tgt: int):
        super().__init__()
        self.enc_src = CharEncoder(vocab_src)
        self.enc_tgt = CharEncoder(vocab_tgt)

    def forward(self, xs, xt):
        zs = self.enc_src(xs)
        zt = self.enc_tgt(xt)
        return zs, zt


def main() -> int:
    parser = argparse.ArgumentParser(description="Train alignment dual-encoder (train-only, auto negatives)")
    parser.add_argument(
        "--train-csv",
        default=str(WORKSPACE_ROOT / "datasets" / "pa" / "train.csv"),
        help="학습용 train.csv 경로 (기본: datasets/p2s/train.csv)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=0, help="학습 배치 수 상한(0이면 제한 없음)")
    parser.add_argument("--temperature", type=float, default=0.07, help="contrastive temperature")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="재현성 시드(0이면 시드 고정 안 함)",
    )

    parser.add_argument("--enable-hard-neg", action="store_true", help="경계 쉬프트 하드 네거티브 랭킹 손실 추가")
    parser.add_argument("--hard-neg-mode", default="prefix_token", choices=["prefix_token", "full"])
    parser.add_argument("--hard-neg-weight", type=float, default=0.5)
    parser.add_argument("--hard-neg-margin", type=float, default=0.15)

    args = parser.parse_args()

    seed = int(args.seed)
    if seed != 0:
        # Best-effort determinism. Note: some CUDA kernels may still be nondeterministic
        # depending on the environment/driver, but this reduces run-to-run variance a lot.
        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        # CUBLAS determinism (must be set before CUDA context is initialized).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import random

            random.seed(seed)
        except Exception:
            pass
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        try:
            # Prefer failing loudly if a nondeterministic op is used.
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions may not support this; keep going.
            pass

    train_csv = Path(args.train_csv)
    # 안전장치: test/val 파일을 train으로 넣는 사고 방지
    lower = str(train_csv).replace("\\", "/").lower()
    if "/test" in lower or "/val" in lower or lower.endswith("test.csv") or lower.endswith("val.csv"):
        raise SystemExit(f"train-only 학습에서는 test/val 파일을 사용할 수 없습니다: {train_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PaTrainCsvDataset(
        train_csv,
        build_vocab=True,
        max_len=512,
        enable_hard_neg=bool(args.enable_hard_neg),
        hard_neg_mode=str(args.hard_neg_mode),
    )

    loader_generator = None
    if seed != 0:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch),
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )

    model = DualEncoder(vocab_src=len(train_ds.vocab_src), vocab_tgt=len(train_ds.vocab_tgt)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    temperature = float(args.temperature)
    if temperature <= 0:
        raise SystemExit("--temperature는 0보다 커야 합니다")

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_contrast = 0.0
        total_rank = 0.0
        total_rank_n = 0
        steps_ran = 0

        for step_i, (xs, xt, xs_hn, hn) in enumerate(train_loader, 1):
            steps_ran = step_i
            xs = xs.to(device)
            xt = xt.to(device)
            xs_hn = xs_hn.to(device)
            hn = hn.to(device)

            zs, zt = model(xs, xt)

            # in-batch negatives: sim matrix [B,B]
            sim_mat = zs @ zt.t()  # dot product
            logits = sim_mat / temperature
            target = torch.arange(logits.size(0), device=device)
            contrast = nn.functional.cross_entropy(logits, target)

            rank = torch.tensor(0.0, device=device)
            if args.enable_hard_neg:
                mask = hn > 0.5
                if mask.any():
                    zs_hn = model.enc_src(xs_hn)
                    sim_pos = sim_mat.diag()
                    sim_neg = (zs_hn * zt).sum(-1)
                    margin = float(args.hard_neg_margin)
                    diff = sim_pos - sim_neg
                    rank = nn.functional.relu(margin - diff[mask]).mean()
                    total_rank_n += int(mask.sum().item())

            loss = contrast + float(args.hard_neg_weight) * rank

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.detach().item())
            total_contrast += float(contrast.detach().item())
            total_rank += float(rank.detach().item())

            if int(args.max_steps) > 0 and step_i >= int(args.max_steps):
                break

        denom = max(1, steps_ran)
        print(
            f"Epoch {ep}: loss={total_loss/denom:.4f} (contrast={total_contrast/denom:.4f} "
            f"rank={total_rank/denom:.4f} n_rank={total_rank_n})"
        )

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_ROOT / "dual_encoder_alignment_pa.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_src": train_ds.vocab_src,
            "vocab_tgt": train_ds.vocab_tgt,
        },
        out_path,
    )
    print(f"✅ Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
