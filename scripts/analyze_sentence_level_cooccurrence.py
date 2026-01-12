#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""문장 단위 한자-현토 공기 분석 및 다의성 탐지 (개선 버전)

핵심 개선:
1. 문장 단위 공기: 같은 문장 내 모든 한자-현토 관계 (거리 무관)
2. 문장 끝 현토 탐지: 종결 vs 접속/중간 자동 구분
3. 의문사-의문종결 관계 자연스럽게 포착

입력: datasets/pa/train.csv (원문 컬럼)
출력:
  - 현토별 선행 한자 클러스터 분석
  - 종결/접속 기능별 분리
  - 다의성 자동 탐지

사용 예:
    python scripts/analyze_sentence_level_cooccurrence.py \
        --csv datasets/pa/train.csv \
        --out-dir hyeonto/reports/sentence_level_analysis
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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


def extract_sentence_features(sentence: str) -> list[dict]:
    """문장에서 (한자, 현토, 위치정보) 추출
    
    Returns:
        list of {
            'hanja': str,
            'marker': str,
            'is_final': bool,  # 현토가 문장 끝인가?
            'position': float  # 0~1 사이 상대 위치
        }
    """
    if not sentence or str(sentence) == "nan":
        return []
    
    sentence = str(sentence).strip()
    if not sentence:
        return []
    
    # 모든 한자와 위치
    hanja_positions = []
    for i, char in enumerate(sentence):
        if re.match(r'\p{Han}', char):
            hanja_positions.append((i, char))
    
    # 현토 패턴: 한자 뒤에 오는 한글
    marker_pattern = re.compile(r'(\p{Han})(\p{Hangul}+)')
    
    results = []
    sentence_len = len(sentence)
    
    for m in marker_pattern.finditer(sentence):
        last_hanja = m.group(1)
        marker = normalize_marker(m.group(2))
        marker_end = m.end()
        
        # 문장 끝 여부 (마지막 3글자 이내)
        is_final = (sentence_len - marker_end) <= 3
        
        # 상대 위치
        position = marker_end / sentence_len if sentence_len > 0 else 0
        
        # 같은 문장 내 모든 한자와 연결 (문장 단위 공기)
        for hanja_pos, hanja in hanja_positions:
            if hanja_pos < m.start():  # 현토 앞의 한자만
                results.append({
                    'hanja': hanja,
                    'marker': marker,
                    'is_final': is_final,
                    'position': position,
                })
    
    return results


def ppmi_transform(C: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    total = C.sum()
    if total <= 0:
        return np.zeros_like(C)
    p_ij = C / (total + eps)
    p_i = C.sum(axis=1, keepdims=True) / (total + eps)
    p_j = C.sum(axis=0, keepdims=True) / (total + eps)
    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
    return np.maximum(pmi, 0.0).astype(np.float32)


def find_optimal_k(X: np.ndarray, max_k: int = 5, min_samples: int = 10) -> tuple[int, float]:
    if len(X) < min_samples:
        return 1, 0.0
    max_k = min(max_k, len(X) - 1)
    if max_k < 2:
        return 1, 0.0
    
    best_k, best_score = 1, -1
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
    
    return (best_k, best_score) if best_score >= 0.25 else (1, best_score)


def main() -> int:
    p = argparse.ArgumentParser(description="문장 단위 한자-현토 공기 분석")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/sentence_level_analysis"))
    p.add_argument("--src-col", type=str, default="원문")
    p.add_argument("--top-markers", type=int, default=30)
    p.add_argument("--min-hanja-count", type=int, default=10)
    p.add_argument("--svd-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.csv.exists():
        print(f"❌ 파일 없음: {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📄 로드: {len(df):,}개 문장")

    # 1. 문장별 feature 추출
    print("📊 문장별 한자-현토 관계 추출 중...")
    
    # 공기 카운트 (전체 / 종결 / 비종결)
    hanja_marker_all: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    hanja_marker_final: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    hanja_marker_nonfinal: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    marker_total: dict[str, float] = defaultdict(float)
    marker_final_count: dict[str, int] = defaultdict(int)
    marker_nonfinal_count: dict[str, int] = defaultdict(int)
    hanja_total: dict[str, float] = defaultdict(float)

    for _, row in df.iterrows():
        sentence = row.get(args.src_col, "")
        features = extract_sentence_features(sentence)
        
        for feat in features:
            h, m = feat['hanja'], feat['marker']
            is_final = feat['is_final']
            
            hanja_marker_all[h][m] += 1
            marker_total[m] += 1
            hanja_total[h] += 1
            
            if is_final:
                hanja_marker_final[h][m] += 1
                marker_final_count[m] += 1
            else:
                hanja_marker_nonfinal[h][m] += 1
                marker_nonfinal_count[m] += 1

    # 2. 필터링
    top_markers = sorted(marker_total.keys(), key=lambda m: -marker_total[m])[:args.top_markers]
    valid_hanja = [h for h, cnt in hanja_total.items() if cnt >= args.min_hanja_count]
    
    print(f"✅ 분석 대상: {len(valid_hanja)}개 한자, {len(top_markers)}개 현토")

    # 3. 공기행렬 구성
    hanja_list = sorted(valid_hanja)
    marker_list = sorted(top_markers)
    hanja_to_idx = {h: i for i, h in enumerate(hanja_list)}
    marker_to_idx = {m: j for j, m in enumerate(marker_list)}

    C = np.zeros((len(hanja_list), len(marker_list)), dtype=np.float32)
    C_final = np.zeros_like(C)
    C_nonfinal = np.zeros_like(C)
    
    for h in hanja_list:
        for m in marker_list:
            C[hanja_to_idx[h], marker_to_idx[m]] = hanja_marker_all[h].get(m, 0)
            C_final[hanja_to_idx[h], marker_to_idx[m]] = hanja_marker_final[h].get(m, 0)
            C_nonfinal[hanja_to_idx[h], marker_to_idx[m]] = hanja_marker_nonfinal[h].get(m, 0)

    print(f"✅ 공기행렬: {C.shape}, nnz={(C > 0).sum():,}")

    # 4. SVD 임베딩
    X = ppmi_transform(C)
    svd_dim = min(args.svd_dim, min(X.shape) - 1)
    svd = TruncatedSVD(n_components=svd_dim, random_state=args.seed)
    hanja_emb = svd.fit_transform(X)
    
    print(f"✅ 한자 임베딩: {hanja_emb.shape}")

    # 5. 현토별 분석 (종결/비종결 분리)
    analysis_results = []
    
    for m_idx, marker in enumerate(marker_list):
        total = int(marker_total[marker])
        final_cnt = marker_final_count[marker]
        nonfinal_cnt = marker_nonfinal_count[marker]
        final_ratio = final_cnt / total if total > 0 else 0
        
        # 종결 비율로 기능 추정
        if final_ratio > 0.7:
            function_type = "종결"
        elif final_ratio < 0.3:
            function_type = "접속/중간"
        else:
            function_type = "혼합 (다의적?)"
        
        # 종결/비종결 각각의 한자 분포 비교
        final_hanja = [(h, C_final[hanja_to_idx[h], m_idx]) 
                       for h in hanja_list if C_final[hanja_to_idx[h], m_idx] > 0]
        nonfinal_hanja = [(h, C_nonfinal[hanja_to_idx[h], m_idx]) 
                          for h in hanja_list if C_nonfinal[hanja_to_idx[h], m_idx] > 0]
        
        # 종결에서만 나타나는 상위 한자 (의문사 후보?)
        final_only = sorted(final_hanja, key=lambda x: -x[1])[:10]
        final_only_str = ", ".join([f"{h}({int(c)})" for h, c in final_only[:5]])
        
        analysis_results.append({
            "marker": marker,
            "total_count": total,
            "final_count": final_cnt,
            "nonfinal_count": nonfinal_cnt,
            "final_ratio": round(final_ratio, 3),
            "function_type": function_type,
            "top_final_hanja": final_only_str,
        })

    # 6. 다의성 탐지 (클러스터링)
    polysemy_results = []
    
    for m_idx, marker in enumerate(marker_list):
        hanja_counts = [(hanja_to_idx[h], h, C[hanja_to_idx[h], m_idx]) 
                        for h in hanja_list if C[hanja_to_idx[h], m_idx] > 0]
        
        if len(hanja_counts) < 15:
            polysemy_results.append({
                "marker": marker,
                "optimal_k": 1,
                "silhouette": 0.0,
                "polysemy": "데이터부족",
            })
            continue
        
        top_hanja = sorted(hanja_counts, key=lambda x: -x[2])[:100]
        indices = [x[0] for x in top_hanja]
        subset_emb = hanja_emb[indices]
        
        opt_k, sil_score = find_optimal_k(subset_emb, max_k=4)
        
        # 클러스터 정보
        cluster_info = ""
        if opt_k >= 2:
            kmeans = KMeans(n_clusters=opt_k, random_state=args.seed, n_init=10)
            labels = kmeans.fit_predict(subset_emb)
            
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                clusters[label].append((top_hanja[i][1], int(top_hanja[i][2])))
            
            parts = []
            for cl in sorted(clusters.keys()):
                samples = sorted(clusters[cl], key=lambda x: -x[1])[:5]
                parts.append(f"C{cl}:[{','.join([h for h,c in samples])}]")
            cluster_info = " | ".join(parts)
        
        polysemy_results.append({
            "marker": marker,
            "optimal_k": opt_k,
            "silhouette": round(sil_score, 3),
            "polysemy": "⚠️다의적" if opt_k >= 2 and sil_score >= 0.25 else "단일",
            "cluster_info": cluster_info,
        })

    # 7. 결과 저장
    # 기능 분석 CSV
    func_df = pd.DataFrame(analysis_results)
    func_df.to_csv(args.out_dir / "marker_function_analysis.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 저장: marker_function_analysis.csv")
    
    # 다의성 분석 CSV
    poly_df = pd.DataFrame(polysemy_results)
    poly_df = poly_df.sort_values("silhouette", ascending=False)
    poly_df.to_csv(args.out_dir / "marker_polysemy_analysis.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 저장: marker_polysemy_analysis.csv")
    
    # 시각화
    try:
        import plotly.graph_objects as go
        
        # 종결 비율 바차트
        func_df_sorted = func_df.sort_values("final_ratio", ascending=True)
        colors = ["#d62728" if r < 0.3 else ("#2ca02c" if r > 0.7 else "#ff7f0e") 
                  for r in func_df_sorted["final_ratio"]]
        
        fig = go.Figure(go.Bar(
            x=func_df_sorted["final_ratio"],
            y=func_df_sorted["marker"],
            orientation="h",
            marker_color=colors,
            text=[f'{r:.0%}' for r in func_df_sorted["final_ratio"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="현토별 문장 종결 위치 비율",
            xaxis_title="종결 비율 (0=항상 중간, 1=항상 끝)",
            yaxis_title="현토",
            height=max(500, len(top_markers) * 22),
            width=800,
        )
        fig.write_html(str(args.out_dir / "marker_final_ratio.html"), include_plotlyjs="cdn")
        print(f"✅ 저장: marker_final_ratio.html")
        
    except ImportError:
        print("⚠️ plotly 미설치")

    # 요약 출력
    print("\n📊 기능 분석 (종결 비율 기준):")
    print("-" * 60)
    for r in sorted(analysis_results, key=lambda x: -x["final_ratio"])[:10]:
        print(f"  {r['marker']:>6} : {r['final_ratio']:.0%} 종결, {r['function_type']}")
    
    print("\n📊 다의성 분석:")
    print("-" * 60)
    for r in sorted(polysemy_results, key=lambda x: -x["silhouette"])[:10]:
        print(f"  {r['marker']:>6} : K={r['optimal_k']}, Sil={r['silhouette']:.2f}, {r['polysemy']}")

    # Config
    cfg = {
        "csv": str(args.csv),
        "top_markers": args.top_markers,
        "min_hanja_count": args.min_hanja_count,
        "num_sentences": len(df),
        "num_hanja": len(hanja_list),
    }
    (args.out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 전체 결과: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
