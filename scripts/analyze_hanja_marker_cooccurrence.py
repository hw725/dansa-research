#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""한자-현토 공기 임베딩 분석 및 통사적 다의성 자동 탐지

목표:
1. 한자-현토 공기행렬로 한자 임베딩 생성
2. 각 현토 앞에 오는 한자들의 분포 분석
3. 다의성 자동 탐지: 선행 한자가 여러 클러스터로 분리되면 다의적

출력:
  - 한자 임베딩 (SVD)
  - 현토별 선행 한자 클러스터 분석
  - 다의성 후보 목록

사용 예:
    python scripts/analyze_hanja_marker_cooccurrence.py \
        --csv hyeonto/reports/recluster_k16_child/reclustered.csv \
        --out-dir hyeonto/reports/hanja_marker_analysis \
        --top-markers 30
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import regex as re
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 한자 + 현토 추출 정규식
_CJK_MARKER_RE = re.compile(r"(?P<cjk>\p{Han}+)(?P<marker>\p{Hangul}+)?")

# 현토 정규화
HYEONTO_REPLACE_MAP = {
    "은": "는", "이": "가", "을": "를", "과": "와", "ㅣ": "가",
}


def normalize_marker(marker: str) -> str:
    if not marker:
        return marker
    if marker in HYEONTO_REPLACE_MAP:
        return HYEONTO_REPLACE_MAP[marker]
    if len(marker) > 1 and marker[0] in ("이", "으"):
        return marker[1:]
    return marker


def extract_hanja_marker_pairs(text: object, use_distance_weight: bool = True) -> list[tuple[str, str, float]]:
    """텍스트에서 (한자, 현토, 가중치) 삼중쌍 추출
    
    구 내 모든 한자와 현토의 관계를 거리 가중치와 함께 추출
    예: '何人이來고' → [('何','고',0.25), ('人','고',0.5), ('來','고',1.0)]
    
    가중치: 1 / (distance), 직전 한자가 가장 높음
    """
    if text is None or str(text) == "nan":
        return []
    
    text_str = str(text)
    pairs = []
    
    # 모든 한자 위치와 글자 추출
    hanja_positions = []  # [(position, char), ...]
    for i, char in enumerate(text_str):
        if re.match(r'\p{Han}', char):
            hanja_positions.append((i, char))
    
    # 모든 현토와 그 위치 추출
    for m in _CJK_MARKER_RE.finditer(text_str):
        marker = m.group("marker")
        if not marker:
            continue
        marker = normalize_marker(marker)
        marker_start = m.start("marker") if m.group("marker") else m.end()
        
        # 현토 앞의 모든 한자와 쌍 생성
        for pos, hanja in hanja_positions:
            if pos < marker_start:
                distance = marker_start - pos
                if use_distance_weight:
                    # 거리 가중치: 1/sqrt(distance) 또는 1/distance
                    # sqrt 사용하면 먼 거리도 어느 정도 반영
                    weight = 1.0 / (distance ** 0.5)
                else:
                    weight = 1.0
                pairs.append((hanja, marker, weight))
    
    return pairs



def ppmi_transform(C: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """PPMI 변환"""
    total = C.sum()
    if total <= 0:
        return np.zeros_like(C)
    p_ij = C / (total + eps)
    p_i = C.sum(axis=1, keepdims=True) / (total + eps)
    p_j = C.sum(axis=0, keepdims=True) / (total + eps)
    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
    return np.maximum(pmi, 0.0).astype(np.float32)


def find_optimal_k(X: np.ndarray, max_k: int = 5, min_samples: int = 10) -> tuple[int, float]:
    """Silhouette score로 최적 클러스터 수 결정"""
    if len(X) < min_samples:
        return 1, 0.0
    
    max_k = min(max_k, len(X) - 1)
    if max_k < 2:
        return 1, 0.0
    
    best_k = 1
    best_score = -1
    
    for k in range(2, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    
    # 유의미한 분리 기준: silhouette > 0.25
    if best_score < 0.25:
        return 1, best_score
    
    return best_k, best_score


def main() -> int:
    p = argparse.ArgumentParser(description="한자-현토 공기 분석 및 다의성 탐지")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/hanja_marker_analysis"))
    p.add_argument("--src-cols", type=str, default="src_left,src_right")
    p.add_argument("--top-markers", type=int, default=30, help="분석할 주요 현토 수")
    p.add_argument("--min-hanja-count", type=int, default=5, help="최소 한자 빈도")
    p.add_argument("--svd-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    src_cols = [c.strip() for c in args.src_cols.split(",") if c.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 로드: {len(df):,}개 행")

    # 1. (한자, 현토, 가중치) 삼중쌍 수집
    hanja_marker_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    marker_total: dict[str, float] = defaultdict(float)
    hanja_total: dict[str, float] = defaultdict(float)

    for _, row in df.iterrows():
        for col in src_cols:
            for hanja, marker, weight in extract_hanja_marker_pairs(row.get(col)):
                hanja_marker_counts[hanja][marker] += weight
                marker_total[marker] += weight
                hanja_total[hanja] += weight

    # 2. 상위 현토 및 빈도 이상 한자 필터링
    top_markers = sorted(marker_total.keys(), key=lambda m: -marker_total[m])[:args.top_markers]
    valid_hanja = [h for h, cnt in hanja_total.items() if cnt >= args.min_hanja_count]

    print(f"✅ 분석 대상: {len(valid_hanja)}개 한자, {len(top_markers)}개 현토")

    # 3. 공기행렬 구성 (한자 × 현토)
    hanja_list = sorted(valid_hanja)
    marker_list = sorted(top_markers)
    hanja_to_idx = {h: i for i, h in enumerate(hanja_list)}
    marker_to_idx = {m: j for j, m in enumerate(marker_list)}

    C = np.zeros((len(hanja_list), len(marker_list)), dtype=np.float32)
    for h in hanja_list:
        for m in marker_list:
            C[hanja_to_idx[h], marker_to_idx[m]] = hanja_marker_counts[h].get(m, 0)

    print(f"✅ 공기행렬: {C.shape}, nnz={(C > 0).sum():,}")

    # 4. PPMI + SVD로 한자 임베딩 생성
    X = ppmi_transform(C)
    svd_dim = min(args.svd_dim, min(X.shape) - 1)
    svd = TruncatedSVD(n_components=svd_dim, random_state=args.seed)
    hanja_emb = svd.fit_transform(X)  # (H, k)
    marker_emb = (svd.components_.T * svd.singular_values_).astype(np.float32)  # (M, k)

    print(f"✅ 임베딩 생성: 한자 {hanja_emb.shape}, 현토 {marker_emb.shape}")

    # 5. 현토별 다의성 분석
    polysemy_results = []

    for m_idx, marker in enumerate(marker_list):
        # 이 현토 앞에 자주 오는 한자들 (빈도 기준)
        hanja_counts_for_marker = []
        for h_idx, hanja in enumerate(hanja_list):
            cnt = C[h_idx, m_idx]
            if cnt > 0:
                hanja_counts_for_marker.append((h_idx, hanja, cnt))
        
        if len(hanja_counts_for_marker) < 10:
            polysemy_results.append({
                "marker": marker,
                "total_count": int(marker_total[marker]),
                "unique_hanja": len(hanja_counts_for_marker),
                "optimal_k": 1,
                "silhouette": 0.0,
                "polysemy": "데이터부족",
                "cluster_info": "",
            })
            continue
        
        # 상위 N개 한자만 분석
        top_hanja = sorted(hanja_counts_for_marker, key=lambda x: -x[2])[:100]
        indices = [x[0] for x in top_hanja]
        hanja_subset_emb = hanja_emb[indices]
        
        # 최적 클러스터 수 탐색
        opt_k, sil_score = find_optimal_k(hanja_subset_emb, max_k=4)
        
        # 클러스터별 대표 한자 추출
        cluster_info = ""
        if opt_k >= 2:
            kmeans = KMeans(n_clusters=opt_k, random_state=args.seed, n_init=10)
            labels = kmeans.fit_predict(hanja_subset_emb)
            
            cluster_samples = defaultdict(list)
            for i, label in enumerate(labels):
                h_idx, hanja, cnt = top_hanja[i]
                cluster_samples[label].append((hanja, int(cnt)))
            
            parts = []
            for cl in sorted(cluster_samples.keys()):
                samples = sorted(cluster_samples[cl], key=lambda x: -x[1])[:5]
                sample_str = ",".join([f"{h}({c})" for h, c in samples])
                parts.append(f"C{cl}:[{sample_str}]")
            cluster_info = " | ".join(parts)
        
        polysemy_results.append({
            "marker": marker,
            "total_count": int(marker_total[marker]),
            "unique_hanja": len(hanja_counts_for_marker),
            "optimal_k": opt_k,
            "silhouette": round(sil_score, 3),
            "polysemy": "⚠️다의적" if opt_k >= 2 and sil_score >= 0.25 else "단일",
            "cluster_info": cluster_info,
        })

    # 6. 결과 저장
    # 다의성 분석 CSV
    poly_df = pd.DataFrame(polysemy_results)
    poly_df = poly_df.sort_values("silhouette", ascending=False)
    poly_df.to_csv(args.out_dir / "marker_polysemy_analysis.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 저장: marker_polysemy_analysis.csv")

    # 한자 임베딩 CSV
    hanja_emb_df = pd.DataFrame(hanja_emb, index=hanja_list)
    hanja_emb_df.insert(0, "count", [hanja_total[h] for h in hanja_list])
    hanja_emb_df.to_csv(args.out_dir / "hanja_embedding.csv", encoding="utf-8-sig")
    print(f"✅ 저장: hanja_embedding.csv")

    # 시각화 (상위 다의적 현토)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 다의적 현토 상위 5개에 대해 한자 분포 시각화
        polysemy_markers = [r for r in polysemy_results if r["polysemy"] == "⚠️다의적"][:5]
        
        if polysemy_markers:
            # 2D PCA로 축소
            pca = PCA(n_components=2, random_state=args.seed)
            hanja_2d = pca.fit_transform(hanja_emb)
            
            for pm in polysemy_markers:
                marker = pm["marker"]
                m_idx = marker_to_idx[marker]
                
                # 이 현토와 공기하는 한자들
                hanja_for_viz = []
                for h_idx, hanja in enumerate(hanja_list):
                    cnt = C[h_idx, m_idx]
                    if cnt > 0:
                        hanja_for_viz.append((h_idx, hanja, cnt))
                
                top_hanja = sorted(hanja_for_viz, key=lambda x: -x[2])[:80]
                indices = [x[0] for x in top_hanja]
                
                # 클러스터링
                subset_emb = hanja_emb[indices]
                kmeans = KMeans(n_clusters=pm["optimal_k"], random_state=args.seed, n_init=10)
                labels = kmeans.fit_predict(subset_emb)
                
                # 시각화
                fig = go.Figure()
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                
                for cl in range(pm["optimal_k"]):
                    cl_indices = [i for i, l in enumerate(labels) if l == cl]
                    cl_hanja = [top_hanja[i][1] for i in cl_indices]
                    cl_counts = [top_hanja[i][2] for i in cl_indices]
                    cl_x = [hanja_2d[top_hanja[i][0], 0] for i in cl_indices]
                    cl_y = [hanja_2d[top_hanja[i][0], 1] for i in cl_indices]
                    
                    fig.add_trace(go.Scatter(
                        x=cl_x, y=cl_y,
                        mode="markers+text",
                        marker=dict(size=np.clip(np.log1p(cl_counts) * 3, 6, 20), color=colors[cl % len(colors)], opacity=0.7),
                        text=cl_hanja,
                        textposition="top center",
                        textfont=dict(size=9),
                        name=f"Cluster {cl}",
                        hovertemplate="%{text}<br>count=%{customdata}<extra></extra>",
                        customdata=cl_counts,
                    ))
                
                fig.update_layout(
                    title=f"현토 '{marker}' 선행 한자 분포 (K={pm['optimal_k']}, Sil={pm['silhouette']:.2f})",
                    xaxis_title="PCA 1",
                    yaxis_title="PCA 2",
                    width=900,
                    height=700,
                )
                fig.write_html(str(args.out_dir / f"polysemy_{marker}.html"), include_plotlyjs="cdn")
                print(f"✅ 저장: polysemy_{marker}.html")

    except ImportError:
        print("⚠️ plotly 미설치 - 시각화 생략")

    # 요약 출력
    print("\n📊 다의성 분석 결과:")
    print("-" * 70)
    for r in sorted(polysemy_results, key=lambda x: -x["silhouette"])[:15]:
        print(f"  {r['marker']:>6} : K={r['optimal_k']}, Sil={r['silhouette']:.2f}, {r['polysemy']}")
        if r["cluster_info"]:
            print(f"         {r['cluster_info'][:80]}...")

    # Config
    cfg = {
        "csv": str(args.csv),
        "top_markers": args.top_markers,
        "min_hanja_count": args.min_hanja_count,
        "svd_dim": svd_dim,
        "num_hanja": len(hanja_list),
        "polysemy_detected": sum(1 for r in polysemy_results if r["polysemy"] == "⚠️다의적"),
    }
    (args.out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 전체 결과: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
