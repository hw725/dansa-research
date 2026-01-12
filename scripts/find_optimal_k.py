#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""최적 클러스터 수(K) 탐색 및 임베딩 캐시 저장 스크립트

Elbow method, Silhouette score, Calinski-Harabasz, Davies-Bouldin 지표를 
사용하여 최적의 K값을 도출합니다.

임베딩 결과를 .npy 파일로 캐시하여 재사용할 수 있습니다.

출력:
- optimal_k_analysis.md: K값 분석 리포트
- optimal_k_plot.html: 시각화 (Plotly)
- embeddings_cache.npy: 임베딩 캐시 (선택)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import sys
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


@dataclass
class BoundaryInstance:
    book_name: str
    paragraph_id: int
    left_sentence_id: int
    right_sentence_id: int
    src_left: str
    src_right: str
    tgt_left: str
    tgt_right: str

    def to_embed_text(self, use_src: bool = True, use_tgt: bool = True) -> str:
        parts = []
        if use_src:
            parts.append(f"[SRC_L]{self.src_left}[SRC_R]{self.src_right}")
        if use_tgt:
            parts.append(f"[TGT_L]{self.tgt_left}[TGT_R]{self.tgt_right}")
        return " ".join(parts)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def compute_embeddings_batched(texts: List[str], batch_size: int, device_id: Optional[int], resume_path: Optional[Path] = None) -> np.ndarray:
    from FlagEmbedding import BGEM3FlagModel
    from tqdm import tqdm
    
    existing_vecs = []
    start_idx = 0
    if resume_path and resume_path.exists():
        try:
            existing_vecs = [np.load(str(resume_path))]
            start_idx = existing_vecs[0].shape[0]
            print(f"  >>> 이어하기 발견: {start_idx}건부터 시작합니다.", flush=True)
        except Exception as e:
            print(f"  >>> 이어하기 로드 실패 (무시하고 새로 시작): {e}", flush=True)

    if start_idx >= len(texts):
        return existing_vecs[0]

    print(f"  모델 로딩 중 (device: {device_id})...", flush=True)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=f"cuda:{device_id}" if device_id is not None else "cpu")
    
    all_vecs = existing_vecs
    total = len(texts)
    pbar = tqdm(total=total, desc="  임베딩 계산")
    pbar.update(start_idx)
    
    for i in range(start_idx, total, batch_size):
        batch = texts[i:i+batch_size]
        out = model.encode(batch, batch_size=batch_size, max_length=512)
        all_vecs.append(np.array(out["dense_vecs"], dtype=np.float32))
        pbar.update(len(batch))
        
        # 주기적 중간 저장 (약 12800건마다) 및 로그
        if (i // batch_size) % 10 == 0:
            print(f"  >>> progress: {min(i+batch_size, total)}/{total} ({min(i+batch_size, total)/total*100:.1f}%)", flush=True)
            if resume_path:
                temp_X = np.vstack(all_vecs)
                np.save(str(resume_path), temp_X)
                
    pbar.close()
    return np.vstack(all_vecs)


def load_pa_sentence_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def _safe_int(x: object, default: int = -1) -> int:
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def iter_boundary_instances(df: pd.DataFrame, analysis_type: str = "PA"):
    if analysis_type == "PA":
        gcols = ["book_name", "문단식별자"]
        scol = "문장식별자"
    else:  # SA
        gcols = ["book_name", "문장식별자"]
        scol = "구식별자"

    df = df.sort_values(by=gcols + [scol])
    for group_key, grp in df.groupby(gcols, sort=False):
        rows = grp.to_dict("records")
        for i in range(len(rows) - 1):
            left_row = rows[i]
            right_row = rows[i + 1]
            yield BoundaryInstance(
                book_name=str(left_row.get("book_name", group_key[0])),
                paragraph_id=_safe_int(left_row.get(gcols[1], group_key[1])),
                left_sentence_id=_safe_int(left_row.get(scol, -1)),
                right_sentence_id=_safe_int(rows[i+1].get(scol, -1)),
                src_left=str(left_row.get("원문", "")),
                src_right=str(right_row.get("원문", "")),
                tgt_left=str(left_row.get("번역문", "")),
                tgt_right=str(right_row.get("번역문", "")),
            )


def find_optimal_k(
    X: np.ndarray,
    k_min: int = 4,
    k_max: int = 32,
    k_step: int = 2
) -> Tuple[int, Dict]:
    """여러 지표를 사용하여 최적 K값 탐색"""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    k_range = list(range(k_min, k_max + 1, k_step))
    
    results = {
        "k_values": k_range,
        "inertias": [],
        "silhouette_scores": [],
        "calinski_harabasz_scores": [],
        "davies_bouldin_scores": [],
    }
    
    print(f"최적 K 탐색: {k_min} ~ {k_max} (step={k_step})", flush=True)
    
    for k in k_range:
        print(f"  K={k} 클러스터링 중...", flush=True)
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init="auto")
        labels = km.fit_predict(X)
        
        results["inertias"].append(float(km.inertia_))
        # Silhouette score는 O(N^2)이므로 대규모 데이터에서는 샘플링 필수
        sample_size = min(len(X), 20000)
        results["silhouette_scores"].append(float(silhouette_score(X, labels, sample_size=sample_size, random_state=42)))
        results["calinski_harabasz_scores"].append(float(calinski_harabasz_score(X, labels)))
        results["davies_bouldin_scores"].append(float(davies_bouldin_score(X, labels)))
    
    # 최적 K 결정 (복합 점수 기반)
    optimal_k = determine_optimal_k(results)
    
    return optimal_k, results


def determine_optimal_k(results: Dict) -> int:
    """여러 지표를 종합하여 최적 K 결정"""
    k_values = results["k_values"]
    
    # 각 지표별 최적점 계산
    # Silhouette: 높을수록 좋음
    sil_best_idx = np.argmax(results["silhouette_scores"])
    
    # Calinski-Harabasz: 높을수록 좋음
    ch_best_idx = np.argmax(results["calinski_harabasz_scores"])
    
    # Davies-Bouldin: 낮을수록 좋음
    db_best_idx = np.argmin(results["davies_bouldin_scores"])
    
    # Elbow point (Inertia의 2차 미분)
    elbow_idx = find_elbow_point(results["inertias"])
    
    # 투표 방식으로 최적 K 결정
    votes = [k_values[sil_best_idx], k_values[ch_best_idx], k_values[db_best_idx], k_values[elbow_idx]]
    
    # 가장 많이 투표된 K 또는 중앙값
    from collections import Counter
    vote_counts = Counter(votes)
    optimal_k = vote_counts.most_common(1)[0][0]
    
    print(f"\n=== 최적 K 분석 결과 ===")
    print(f"  Silhouette 최적: K={k_values[sil_best_idx]} (score={results['silhouette_scores'][sil_best_idx]:.4f})")
    print(f"  Calinski-Harabasz 최적: K={k_values[ch_best_idx]}")
    print(f"  Davies-Bouldin 최적: K={k_values[db_best_idx]}")
    print(f"  Elbow point: K={k_values[elbow_idx]}")
    print(f"  >>> 최종 권장: K={optimal_k}")
    
    return optimal_k


def find_elbow_point(inertias: List[float]) -> int:
    """Elbow point 탐지 (2차 미분 기반)"""
    if len(inertias) < 3:
        return 0
    
    # 1차 미분
    first_diff = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    
    # 2차 미분
    second_diff = [first_diff[i] - first_diff[i+1] for i in range(len(first_diff)-1)]
    
    if second_diff:
        return np.argmax(second_diff) + 1
    
    return 0


def generate_optimal_k_plot(results: Dict, out_path: Path) -> None:
    """최적 K 분석 결과 시각화 (Plotly)"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Elbow Method (Inertia)", "Silhouette Score", 
                       "Calinski-Harabasz Index", "Davies-Bouldin Index"]
    )
    
    k_values = results["k_values"]
    
    # Inertia (Elbow)
    fig.add_trace(
        go.Scatter(x=k_values, y=results["inertias"], mode="lines+markers", name="Inertia"),
        row=1, col=1
    )
    
    # Silhouette
    fig.add_trace(
        go.Scatter(x=k_values, y=results["silhouette_scores"], mode="lines+markers", 
                   name="Silhouette", line=dict(color="green")),
        row=1, col=2
    )
    
    # Calinski-Harabasz
    fig.add_trace(
        go.Scatter(x=k_values, y=results["calinski_harabasz_scores"], mode="lines+markers",
                   name="Calinski-Harabasz", line=dict(color="orange")),
        row=2, col=1
    )
    
    # Davies-Bouldin
    fig.add_trace(
        go.Scatter(x=k_values, y=results["davies_bouldin_scores"], mode="lines+markers",
                   name="Davies-Bouldin", line=dict(color="red")),
        row=2, col=2
    )
    
    fig.update_layout(
        title="최적 클러스터 수(K) 분석",
        height=700,
        showlegend=False
    )
    
    fig.write_html(str(out_path))


def write_optimal_k_report(
    out_path: Path,
    optimal_k: int,
    results: Dict,
    analysis_type: str
) -> None:
    """최적 K 분석 리포트 작성"""
    
    lines = [
        f"# {analysis_type} 최적 클러스터 수(K) 분석 리포트",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**탐색 범위**: K={results['k_values'][0]} ~ {results['k_values'][-1]}",
        "",
        "---",
        "",
        "## 분석 결과 요약",
        "",
        f"**권장 K값: {optimal_k}**",
        "",
        "---",
        "",
        "## 지표별 상세",
        "",
        "| K | Inertia | Silhouette | Calinski-Harabasz | Davies-Bouldin |",
        "|:---:|---:|---:|---:|---:|",
    ]
    
    for i, k in enumerate(results["k_values"]):
        lines.append(
            f"| {k} | {results['inertias'][i]:.0f} | "
            f"{results['silhouette_scores'][i]:.4f} | "
            f"{results['calinski_harabasz_scores'][i]:.0f} | "
            f"{results['davies_bouldin_scores'][i]:.4f} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## 지표 해석",
        "",
        "| 지표 | 해석 | 최적 기준 |",
        "|:---|:---|:---|",
        "| **Inertia** | 클러스터 내 분산 합계 | Elbow point |",
        "| **Silhouette** | 클러스터 응집도 vs 분리도 | 높을수록 좋음 (max=1.0) |",
        "| **Calinski-Harabasz** | 클러스터 간 분산 / 클러스터 내 분산 | 높을수록 좋음 |",
        "| **Davies-Bouldin** | 클러스터 간 유사도 | 낮을수록 좋음 |",
        "",
        "---",
        "",
        f"**분석 완료**: 권장 K={optimal_k}",
    ])
    
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="최적 K값 탐색 및 임베딩 캐시")
    p.add_argument("--input", type=str, required=True, help="입력 CSV")
    p.add_argument("--out-dir", type=str, required=True, help="출력 디렉토리")
    p.add_argument("--analysis-type", type=str, default="PA", help="분석 타입 (PA/SA)")
    p.add_argument("--max-boundaries", type=int, default=500000, help="최대 경계 샘플 수")
    
    # K값 탐색 범위
    p.add_argument("--k-min", type=int, default=4, help="최소 K")
    p.add_argument("--k-max", type=int, default=32, help="최대 K")
    p.add_argument("--k-step", type=int, default=2, help="K 증가 단위")
    
    # 임베딩 캐시
    p.add_argument("--load-embeddings", type=str, default=None, help="임베딩 캐시 로드 경로 (.npy)")
    p.add_argument("--save-embeddings", type=str, default=None, help="임베딩 캐시 저장 경로 (.npy)")
    
    # GPU
    p.add_argument("--device-id", type=int, default=None, help="GPU ID")
    p.add_argument("--batch", type=int, default=128, help="임베딩 배치 크기")
    
    args = p.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 데이터 로드 또는 임베딩 캐시 로드
    if args.load_embeddings and Path(args.load_embeddings).exists():
        print(f"[1/3] 임베딩 캐시 로드: {args.load_embeddings}", flush=True)
        X = np.load(args.load_embeddings)
        print(f"  -> {X.shape[0]}개 임베딩 로드", flush=True)
    else:
        print(f"[1/3] 데이터 로딩 및 임베딩 계산: {args.input}", flush=True)
        df = pd.read_csv(Path(args.input), encoding="utf-8-sig")
        
        instances: List[BoundaryInstance] = []
        for inst in iter_boundary_instances(df, analysis_type=args.analysis_type):
            instances.append(inst)
            if len(instances) >= args.max_boundaries:
                break
        
        print(f"  -> {len(instances)}개 경계 인스턴스 로드", flush=True)
        
        texts = [x.to_embed_text(use_src=True, use_tgt=True) for x in instances]
        resume_cache = Path(args.save_embeddings).with_name(Path(args.save_embeddings).stem + "_resume.npy") if args.save_embeddings else None
        X = compute_embeddings_batched(texts, batch_size=args.batch, device_id=args.device_id, resume_path=resume_cache)
        X = _l2_normalize(X)
        
        # 임베딩 캐시 저장
        if args.save_embeddings:
            save_path = Path(args.save_embeddings)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(save_path), X)
            print(f"  -> 임베딩 캐시 저장: {save_path}")
    
    # 2. 최적 K 탐색
    print(f"[2/3] 최적 K 탐색...")
    optimal_k, results = find_optimal_k(X, k_min=args.k_min, k_max=args.k_max, k_step=args.k_step)
    
    # 3. 결과 저장
    print(f"[3/3] 결과 저장...")
    
    # 리포트
    write_optimal_k_report(
        out_dir / f"optimal_k_analysis_{args.analysis_type.lower()}.md",
        optimal_k, results, args.analysis_type
    )
    
    # 시각화
    generate_optimal_k_plot(results, out_dir / f"optimal_k_plot_{args.analysis_type.lower()}.html")
    
    # JSON 결과
    results["optimal_k"] = optimal_k
    with open(out_dir / f"optimal_k_results_{args.analysis_type.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n완료: {out_dir}")
    print(f"권장 K값: {optimal_k}")


if __name__ == "__main__":
    main()
