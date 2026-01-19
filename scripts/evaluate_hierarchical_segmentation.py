#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
계층적 분할 평가 스크립트
- pd/test → pa 모델로 문장 분할 → p2s/test 정답과 비교 (tgt 기준)
- p2s/test → sa 모델로 구 분할 → s2p/test 정답과 비교 (tgt 기준)
- 멀티태스크 경계 모델(boudary_multitask.pt) 사용
- 결과 저장: test_results/hierarchical/summary.json
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import difflib
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import time
try:
    # 빠른 문자열 유사도
    from rapidfuzz.fuzz import ratio as fuzz_ratio
    _HAVE_RAPIDFUZZ = True
except Exception:
    _HAVE_RAPIDFUZZ = False

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
MODELS_ROOT = WORKSPACE_ROOT / "models"
OUT_ROOT = WORKSPACE_ROOT / "test_results" / "hierarchical"

CHECKPOINT = MODELS_ROOT / "boundary_multitask.pt"

TASK_PA = "pa"
TASK_SA = "sa"
TASK_PD = "pd"


# DualEncoder용 CharEncoder (proj 포함)
class CharEncoderForAlignment(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        # vocab_size는 이미 embedding 크기 (학습 시 len(vocab)+1로 생성됨)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden * 2, 256)

    def forward(self, x):
        e = self.emb(x)
        o, _ = self.lstm(e)
        # mean pool
        m = o.mean(dim=1)
        z = self.proj(m)
        z = nn.functional.normalize(z, dim=-1)
        return z


# Boundary용 CharEncoder (proj 없음)
class CharEncoderForBoundary(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return h


class DualEncoder(nn.Module):
    def __init__(self, vocab_src, vocab_tgt):
        super().__init__()
        self.enc_src = CharEncoderForAlignment(vocab_src)
        self.enc_tgt = CharEncoderForAlignment(vocab_tgt)

    def forward(self, src, tgt):
        v_src = self.enc_src(src)
        v_tgt = self.enc_tgt(tgt)
        return v_src, v_tgt


class AlignmentMatcher:
    """세그먼트 정렬 매칭"""
    def __init__(self, model_path: Path, device: torch.device):
        checkpoint = torch.load(model_path, map_location=device)
        # checkpoint에서 vocab 로드 (학습 시 저장된 것)
        self.vocab_src = checkpoint.get("vocab_src", {})
        self.vocab_tgt = checkpoint.get("vocab_tgt", {})
        self.device = device
        # checkpoint에서 실제 vocab_size 추출
        state_dict = checkpoint.get("state_dict", checkpoint)
        actual_vocab_src = state_dict['enc_src.emb.weight'].shape[0]
        actual_vocab_tgt = state_dict['enc_tgt.emb.weight'].shape[0]
        self.model = DualEncoder(vocab_src=actual_vocab_src, vocab_tgt=actual_vocab_tgt).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"✅ Alignment 모델 로드: {model_path} (vocab_src={len(self.vocab_src)}, vocab_tgt={len(self.vocab_tgt)})")

    def encode_text(self, text: str, is_src: bool = True) -> torch.Tensor:
        vocab = self.vocab_src if is_src else self.vocab_tgt
        # 안전한 인덱스 변환: vocab 범위 체크
        ids = []
        max_idx = max(vocab.values()) if vocab else 0
        for ch in text:
            idx = vocab.get(ch, 0)
            # 범위 벗어나면 0(padding)으로
            if idx > max_idx:
                idx = 0
            ids.append(idx)
        x = torch.tensor([ids], dtype=torch.long).to(self.device)
        return x

    def compute_similarity(self, src_text: str, tgt_text: str) -> float:
        if not src_text or not tgt_text:
            return 0.0
        src_ids = self.encode_text(src_text, is_src=True)
        tgt_ids = self.encode_text(tgt_text, is_src=False)
        with torch.no_grad():
            v_src, v_tgt = self.model(src_ids, tgt_ids)
            # 이미 normalize된 벡터이므로 dot product로 코사인 유사도 계산
            cos_sim = (v_src * v_tgt).sum(dim=-1).item()
        return cos_sim

    def match_segments(self, src_segments: List[str], tgt_segments: List[str]) -> List[str]:
        """greedy matching으로 src_segments를 tgt_segments와 정렬"""
        if not tgt_segments:
            return []
        
        # 각 tgt_segment에 대해 src 구간 찾기
        matched_src = []
        src_idx = 0
        src_str = "".join(src_segments)
        src_positions = []  # 각 segment의 시작 위치
        pos = 0
        for seg in src_segments:
            src_positions.append(pos)
            pos += len(seg)
        src_positions.append(pos)
        
        for tgt_seg in tgt_segments:
            # tgt_seg와 가장 유사한 src 구간 찾기
            best_score = -1
            best_start = src_idx
            best_end = src_idx + 1
            
            # src_idx부터 시작해서 가능한 구간들 평가
            for end_idx in range(src_idx, min(src_idx + 5, len(src_segments) + 1)):
                if end_idx > len(src_segments):
                    break
                src_text = "".join(src_segments[src_idx:end_idx])
                score = self.compute_similarity(src_text, tgt_seg)
                if score > best_score:
                    best_score = score
                    best_start = src_idx
                    best_end = end_idx
            
            # best 구간 추출
            if best_end > best_start:
                matched_src.extend(src_segments[best_start:best_end])
                src_idx = best_end
            else:
                # 실패 시 tgt_seg 길이 비율로 분할
                remaining_src_len = sum(len(s) for s in src_segments[src_idx:])
                ratio = len(tgt_seg) / sum(len(t) for t in tgt_segments) if sum(len(t) for t in tgt_segments) > 0 else 0
                seg_len = int(remaining_src_len * ratio) if ratio > 0 else 1
                
                accum = 0
                seg_start = src_idx
                for i in range(src_idx, len(src_segments)):
                    accum += len(src_segments[i])
                    if accum >= seg_len:
                        matched_src.extend(src_segments[seg_start:i+1])
                        src_idx = i + 1
                        break
                else:
                    matched_src.extend(src_segments[src_idx:])
                    src_idx = len(src_segments)
        
        # 남은 src 세그먼트 추가
        if src_idx < len(src_segments):
            matched_src.extend(src_segments[src_idx:])
        
        return matched_src


# MultiHeadBoundary (boundary tagging)
class MultiHeadBoundary(nn.Module):
    def __init__(self, vocab_size: int, tasks: List[str]):
        super().__init__()
        self.encoder = CharEncoderForBoundary(vocab_size)  # Boundary용 인코더 사용
        hidden_dim = 128
        self.heads = nn.ModuleDict({t: nn.Linear(hidden_dim * 2, 1) for t in tasks})

    def forward(self, x: torch.Tensor, task: str):
        h = self.encoder(x)
        return self.heads[task](h).squeeze(-1)


# -----------------------------
# 유틸
# -----------------------------
def seq_similarity(a: str, b: str) -> float:
    """빠른 문자열 유사도 계산.
    - rapidfuzz 사용 가능 시 fuzz ratio (매우 빠름)
    - 대형 문자열(>2048 chars)은 앞/뒤 샘플링 후 비교로 근사값 계산
    - 완전 일치/빈 문자열 즉시 단축
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAVE_RAPIDFUZZ:
        try:
            return fuzz_ratio(a, b) / 100.0
        except Exception:
            pass
    # fallback: difflib (느림) → 길이 큰 경우 샘플링
    max_chars = 2048
    if len(a) > max_chars or len(b) > max_chars:
        half = max_chars // 2
        a_slice = (a[:half] + a[-half:]) if len(a) > max_chars else a
        b_slice = (b[:half] + b[-half:]) if len(b) > max_chars else b
        return difflib.SequenceMatcher(None, a_slice, b_slice).ratio()
    return difflib.SequenceMatcher(None, a, b).ratio()


def evaluate_segmentation(gt_segments: List[str], pred_segments: List[str], desc: str = "evaluating") -> Dict[str, float]:
    count_match = 1.0 if len(gt_segments) == len(pred_segments) else 0.0
    gt_text = "".join(gt_segments)
    pred_text = "".join(pred_segments)
    # 대형 텍스트 결합 비교는 매우 느리므로 fast similarity 사용
    text_sim = seq_similarity(gt_text, pred_text)
    
    # 세그먼트별 유사도: 최대 길이까지 비교
    max_len = max(len(gt_segments), len(pred_segments))
    sims = []
    for i in tqdm(range(max_len), desc=desc, total=max_len):
        g = gt_segments[i] if i < len(gt_segments) else ""
        p = pred_segments[i] if i < len(pred_segments) else ""
        # 완전 일치 즉시 단축
        if g == p:
            sims.append(1.0)
            continue
        sims.append(seq_similarity(g, p))
    
    avg_seg_sim = sum(sims) / len(sims) if sims else 0.0
    exact = sum(1 for g, p in zip(gt_segments, pred_segments) if g == p)
    exact_rate = exact / max(len(gt_segments), len(pred_segments)) if max(len(gt_segments), len(pred_segments)) > 0 else 0.0
    return {
        "segment_count_gt": len(gt_segments),
        "segment_count_pred": len(pred_segments),
        "segment_count_match": count_match,
        "text_similarity": text_sim,
        "avg_segment_similarity": avg_seg_sim,
        "exact_match_rate": exact_rate,
    }


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    vocab: Dict[str, int] = ckpt["vocab"]
    tasks: List[str] = ckpt.get("tasks", [TASK_PA, TASK_SA, TASK_PD])
    max_len: int = ckpt.get("max_len", 1024)
    model = MultiHeadBoundary(vocab_size=len(vocab) + 1, tasks=tasks).to(device)
    state = ckpt["state_dict"]
    # 구형 체크포인트 호환: emb./lstm. → encoder.emb./encoder.lstm.
    remapped = {}
    for k, v in state.items():
        if k.startswith("emb."):
            remapped["encoder." + k] = v
        elif k.startswith("lstm."):
            remapped["encoder." + k] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped, strict=True)
    model.eval()
    return model, vocab, max_len


def text_to_tensor(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    ids = [vocab.get(ch, 0) for ch in text][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def segment_text(text: str, model: MultiHeadBoundary, vocab: Dict[str, int], max_len: int, task: str, threshold: float = 0.5, device: torch.device = torch.device("cpu")) -> List[str]:
    if not text:
        return []
    
    t0 = time.time()
    x = text_to_tensor(text, vocab, max_len).to(device)
    t1 = time.time()
    
    with torch.no_grad():
        logits = model(x, task)
        probs = torch.sigmoid(logits)[0].cpu().tolist()
    t2 = time.time()
    
    boundaries = [i for i, p in enumerate(probs) if p >= threshold]
    t3 = time.time()
    
    segments = []
    start = 0
    for b in boundaries:
        end = min(b + 1, len(text))
        if end > start:
            segments.append(text[start:end])
            start = end
    if start < len(text):
        segments.append(text[start:])
    t4 = time.time()
    
    # 5초 이상 걸리면 로그
    total = t4 - t0
    if total > 5.0:
        print(f"   ⏱️  slow segment_text: text_len={len(text)} boundaries={len(boundaries)} segs={len(segments)} total={total:.2f}s (tensor={t1-t0:.2f}s model={t2-t1:.2f}s bounds={t3-t2:.2f}s seg={t4-t3:.2f}s)")
    
    return segments


# -----------------------------
# 파이프라인
# -----------------------------
def run_hierarchical(device: str = "cpu", threshold: float = 0.5):
    device_t = torch.device(device)
    if not CHECKPOINT.exists():
        print(f"❌ checkpoint not found: {CHECKPOINT}")
        return

    # 안정적 실행: 진행 로그 추가 및 예외 감싸기
    try:
        model, vocab, max_len = load_checkpoint(CHECKPOINT, device_t)
        print(f"✅ checkpoint loaded: {CHECKPOINT}")
        print(f"   vocab={len(vocab)} max_len={max_len}\n", flush=True)

        # alignment 모델 로드 (checkpoint에서 vocab 자동 로드)
        pa_align_path = MODELS_ROOT / "dual_encoder_alignment_pa.pt"
        sa_align_path = MODELS_ROOT / "dual_encoder_alignment_sa.pt"
        
        pa_matcher = AlignmentMatcher(pa_align_path, device_t) if pa_align_path.exists() else None
        sa_matcher = AlignmentMatcher(sa_align_path, device_t) if sa_align_path.exists() else None
        
        OUT_ROOT.mkdir(parents=True, exist_ok=True)

        # 1) pd → pa 평가 (번역 문장 기준)
        pd_path = DATASETS_ROOT / "pd" / "test.csv"
        pa_path = DATASETS_ROOT / "pa" / "test.csv"
        print(f"📂 loading pd test: {pd_path}", flush=True)
        pd_test = pd.read_csv(pd_path, dtype=str).fillna("")
        print(f"   rows: {len(pd_test)}", flush=True)
        print(f"📂 loading pa test: {pa_path}", flush=True)
        pa_test = pd.read_csv(pa_path, dtype=str).fillna("")
        print(f"   rows: {len(pa_test)}", flush=True)

        # pd의 번역문 분할 → pa의 번역문(정답)과 비교
        # + 각 번역 문장에 맞춰 원문도 재정렬
        print("✂️  segmenting pd→pa (per paragraph)...", flush=True)
        pa_gt_tgt = pa_test["tgt"].tolist()  # 정답: 번역문 문장
        pa_gt_src = pa_test["src"].tolist()  # 정답: 원문 (참고용)
        pa_pred_tgt: List[str] = []
        pa_pred_src: List[str] = []
        
        pd_rows_src = pd_test["src"].tolist()
        pd_rows_tgt = pd_test["tgt"].tolist()
        pbar = tqdm(enumerate(zip(pd_rows_src, pd_rows_tgt), start=1), desc="pd→pa", unit="para", total=len(pd_rows_src))
        for idx, (src, tgt) in pbar:
            pbar.set_postfix({"idx": idx})
            if len(tgt) > 5000:
                print(f"   ⚠️  long text at idx {idx}: len={len(tgt)} chars")
            # 번역문 분할
            tgt_segs = segment_text(tgt, model, vocab, max_len, TASK_PA, threshold, device_t)
            pa_pred_tgt.extend(tgt_segs)
            # 각 번역 세그먼트에 맞춰 원문을 재정렬 (alignment 모델 사용)
            if pa_matcher:
                # 원문을 먼저 초기 분할
                src_init_segs = segment_text(src, model, vocab, max_len, TASK_PA, threshold, device_t)
                # alignment 모델로 매칭
                src_segs = pa_matcher.match_segments(src_init_segs, tgt_segs)
            else:
                # alignment 모델 없으면 길이 비율로 분할
                src_init_segs = segment_text(src, model, vocab, max_len, TASK_PA, threshold, device_t)
                src_segs = src_init_segs
            pa_pred_src.extend(src_segs)
        
        # pd→pa 평가
        print(
            f"📊 evaluating pd→pa (tgt)... gt_segs={len(pa_gt_tgt)} pred_segs={len(pa_pred_tgt)} gt_chars={len(''.join(pa_gt_tgt))} pred_chars={len(''.join(pa_pred_tgt))}",
            flush=True,
        )
        t_eval_start = time.time()
        pd_pa_metrics = evaluate_segmentation(pa_gt_tgt, pa_pred_tgt, desc="eval pd→pa (tgt)")
        print(f"pd→pa done. pred_segments={len(pa_pred_tgt)} elapsed={time.time() - t_eval_start:.1f}s", flush=True)

        # 2) pa → sa 평가 (원문 기준)
        sa_path = DATASETS_ROOT / "sa" / "test.csv"
        print(f"📂 loading sa test: {sa_path}", flush=True)
        sa_test = pd.read_csv(sa_path, dtype=str).fillna("")
        print(f"   rows: {len(sa_test)}", flush=True)

        print("✂️  segmenting pa→sa (per sentence)...", flush=True)
        sa_gt_src = sa_test["src"].tolist()  # 정답: 원문 구
        sa_gt_tgt = sa_test["tgt"].tolist()  # 정답: 번역문 (참고용)
        sa_pred_src: List[str] = []
        sa_pred_tgt: List[str] = []
        
        pa_rows_src = pa_test["src"].tolist()
        pa_rows_tgt = pa_test["tgt"].tolist()
        pbar = tqdm(enumerate(zip(pa_rows_src, pa_rows_tgt), start=1), desc="pa→sa", unit="sent", total=len(pa_rows_src))
        for idx, (src, tgt) in pbar:
            pbar.set_postfix({"idx": idx})
            if len(src) > 5000:
                print(f"   ⚠️  long text at idx {idx}: len={len(src)} chars")
            # 원문 분할
            src_segs = segment_text(src, model, vocab, max_len, TASK_SA, threshold, device_t)
            sa_pred_src.extend(src_segs)
            # 각 원문 세그먼트에 맞춰 번역문을 재정렬 (alignment 모델 사용)
            if sa_matcher:
                # 번역문을 먼저 초기 분할
                tgt_init_segs = segment_text(tgt, model, vocab, max_len, TASK_SA, threshold, device_t)
                # alignment 모델로 매칭
                tgt_segs = sa_matcher.match_segments(tgt_init_segs, src_segs)
            else:
                # alignment 모델 없으면 길이 비율로 분할
                tgt_init_segs = segment_text(tgt, model, vocab, max_len, TASK_SA, threshold, device_t)
                tgt_segs = tgt_init_segs
            sa_pred_tgt.extend(tgt_segs)
        
        print(
            f"📊 evaluating pa→sa (src)... gt_segs={len(sa_gt_src)} pred_segs={len(sa_pred_src)} gt_chars={len(''.join(sa_gt_src))} pred_chars={len(''.join(sa_pred_src))}",
            flush=True,
        )
        t_eval_start = time.time()
        pa_sa_metrics = evaluate_segmentation(sa_gt_src, sa_pred_src, desc="eval pa→sa (src)")
        print(f"pa→sa done. pred_segments={len(sa_pred_src)} elapsed={time.time() - t_eval_start:.1f}s", flush=True)

        summary = {
            "threshold": threshold,
            "checkpoint": str(CHECKPOINT),
            "pd_to_pa": pd_pa_metrics,
            "pa_to_sa": pa_sa_metrics,
            "pd_to_pa_pred_segments": len(pa_pred_tgt),
            "pa_to_sa_pred_segments": len(sa_pred_src),
        }

        # 저장
        out_json = OUT_ROOT / "summary.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 샘플 일부 저장
        sample_path = OUT_ROOT / "samples.txt"
        with sample_path.open("w", encoding="utf-8") as f:
            f.write("[pd→pa] predicted segments (first 20)\n")
            for seg in pa_pred_tgt[:20]:
                f.write(seg + "\n")
            f.write("\n[pa→sa] predicted segments (first 20)\n")
            for seg in sa_pred_src[:20]:
                f.write(seg + "\n")

        print(f"\n=== Hierarchical Segmentation Evaluation ===", flush=True)
        print(f"pd→pa | count_gt={pd_pa_metrics['segment_count_gt']} count_pred={pd_pa_metrics['segment_count_pred']} text_sim={pd_pa_metrics['text_similarity']:.4f} avg_seg_sim={pd_pa_metrics['avg_segment_similarity']:.4f} exact={pd_pa_metrics['exact_match_rate']:.4f}", flush=True)
        print(f"pa→sa | count_gt={pa_sa_metrics['segment_count_gt']} count_pred={pa_sa_metrics['segment_count_pred']} text_sim={pa_sa_metrics['text_similarity']:.4f} avg_seg_sim={pa_sa_metrics['avg_segment_similarity']:.4f} exact={pa_sa_metrics['exact_match_rate']:.4f}", flush=True)
        print(f"\n💾 saving results...", flush=True)
        print(f"Saved: {out_json}", flush=True)
        print(f"Samples: {sample_path}", flush=True)
        print(f"✅ Evaluation complete!", flush=True)

    except Exception as e:
        import traceback
        print("❌ Error during evaluation")
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="계층적 분할 평가 (pd→pa, pa→sa)")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run_hierarchical(device=args.device, threshold=args.threshold)
