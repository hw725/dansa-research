#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""현토 종합 통사 분석 (V6 - src_left/right, tgt_left/right 대응)

분석 레벨:
1. 문장 레벨 (PA): 장거리 한자-현토 공기
2. 구 레벨 (SA): 현토-번역어미 직접 대응
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
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

_CJK_MARKER_RE = re.compile(r"(?P<cjk>\p{Han}+)(?P<marker>\p{Hangul}+)?")

def extract_markers_from_v6_row(row: pd.Series, cols: list[str]) -> list[tuple[str, bool]]:
    """V6 행에서 (현토, 문장끝여부) 추출"""
    results = []
    for col in cols:
        text = str(row.get(col, ""))
        if not text or text == "nan":
            continue
        
        # 문장 끝 판단 (src_right의 끝이면 문장 끝으로 간주)
        is_right_col = "right" in col
        
        found = list(_CJK_MARKER_RE.finditer(text))
        for i, m in enumerate(found):
            marker = m.group("marker")
            if marker:
                norm = normalize_marker(marker)
                # 마지막 컬럼의 마지막 마커면 문장 끝
                is_final = is_right_col and (i == len(found) - 1)
                results.append((norm, is_final))
    return results

def extract_hanja_from_v6_row(row: pd.Series, cols: list[str]) -> list[str]:
    """V6 행에서 한자 추출"""
    results = []
    for col in cols:
        text = str(row.get(col, ""))
        if not text or text == "nan":
            continue
        for char in text:
            if re.match(r'\p{Han}', char):
                results.append(char)
    return results

def extract_translation_ending(tgt_text: str, max_chars: int = 10) -> str:
    if not tgt_text or str(tgt_text) == "nan":
        return ""
    tgt = str(tgt_text).strip()
    # 마지막 어미 부분 추출
    ending = tgt[-max_chars:]
    hangul_only = re.sub(r'[^\p{Hangul}]', '', ending)
    return hangul_only[-5:]

def ppmi_transform(C: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    total = C.sum()
    if total <= 0: return np.zeros_like(C)
    p_ij = C / (total + eps)
    p_i = C.sum(axis=1, keepdims=True) / (total + eps)
    p_j = C.sum(axis=0, keepdims=True) / (total + eps)
    pmi = np.log((p_ij + eps) / (p_i * p_j + eps))
    return np.maximum(pmi, 0.0).astype(np.float32)

def find_optimal_k(X: np.ndarray, max_k: int = 5) -> tuple[int, float]:
    if len(X) < 15: return 1, 0.0
    max_k = min(max_k, len(X) - 1)
    if max_k < 2: return 1, 0.0
    best_k, best_score = 1, -1
    for k in range(2, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except: continue
    return (best_k, best_score) if best_score >= 0.2 else (1, best_score)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pa-csv", type=Path, required=True)
    p.add_argument("--sa-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-count", type=int, default=50)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. PA 로드 및 통계
    pa_df = pd.read_csv(args.pa_csv)
    src_cols = ["src_left", "src_right"]
    
    hanja_marker_counts = defaultdict(lambda: defaultdict(float))
    marker_stats = defaultdict(lambda: {"total": 0, "final": 0})

    print("Analysing PA...")
    for _, row in pa_df.iterrows():
        markers = extract_markers_from_v6_row(row, src_cols)
        hanjas = extract_hanja_from_v6_row(row, src_cols)
        for m, is_final in markers:
            marker_stats[m]["total"] += 1
            if is_final: marker_stats[m]["final"] += 1
            for h in hanjas:
                hanja_marker_counts[h][m] += 1

    valid_markers = [m for m, s in marker_stats.items() if s["total"] >= args.min_count]
    valid_hanja = [h for h, mdict in hanja_marker_counts.items() if sum(mdict.values()) >= 10]
    
    m_list = sorted(valid_markers)
    h_list = sorted(valid_hanja)
    m_idx = {m: i for i, m in enumerate(m_list)}
    h_idx = {h: i for i, h in enumerate(h_list)}

    C = np.zeros((len(h_list), len(m_list)), dtype=np.float32)
    for h, mdict in hanja_marker_counts.items():
        if h in h_idx:
            for m, cnt in mdict.items():
                if m in m_idx:
                    C[h_idx[h], m_idx[m]] = cnt

    X = ppmi_transform(C)
    svd = TruncatedSVD(n_components=min(32, X.shape[1]-1), random_state=42)
    h_emb = svd.fit_transform(X)

    # 2. SA 로드 및 번역 패턴
    sa_df = pd.read_csv(args.sa_csv)
    marker_patterns = defaultdict(Counter)
    
    print("Analysing SA...")
    for _, row in sa_df.iterrows():
        ms = extract_markers_from_v6_row(row, ["src_left", "src_right"])
        ending = extract_translation_ending(str(row.get("tgt_right", "")))
        if not ending: continue
        for m, _ in ms:
            if m in m_idx:
                marker_patterns[m][ending] += 1

    # 3. 종합
    results = []
    for m in m_list:
        s = marker_stats[m]
        f_ratio = s["final"] / s["total"]
        patterns = marker_patterns[m]
        top_p = ", ".join([f"{p}({c})" for p, c in patterns.most_common(3)])
        
        # Silhouette
        m_col = m_idx[m]
        rel_h = [h_idx[h] for h in h_list if C[h_idx[h], m_col] > 0]
        opt_k, sil = 1, 0.0
        if len(rel_h) >= 15:
            opt_k, sil = find_optimal_k(h_emb[rel_h])

        results.append({
            "현토": m,
            "총빈도": s["total"],
            "종결비율": round(f_ratio, 3),
            "번역패턴": top_p,
            "패턴속성": "다의적" if sil > 0.3 else "단일",
            "Silhouette": round(sil, 3),
            "K": opt_k
        })

    rdf = pd.DataFrame(results).sort_values("총빈도", ascending=False)
    rdf.to_csv(args.out_dir / "marker_syntactic_profile.csv", index=False, encoding="utf-8-sig")

    try:
        import plotly.express as px
        fig = px.scatter(rdf, x="종결비율", y="Silhouette", size="총빈도", color="패턴속성", hover_name="현토",
                         title="V6 현토 통사 프로파일 (종결비율 vs 다의성)")
        fig.write_html(str(args.out_dir / "syntactic_profile_scatter.html"), include_plotlyjs="cdn")
    except: pass

    print(f"Done. Saved to {args.out_dir}")

if __name__ == "__main__":
    main()
