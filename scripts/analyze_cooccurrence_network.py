#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""한자-현토 공기(共起) 네트워크 분석 스크립트

특정 한자가 특정 현토 마커와 함께 나타나는 패턴을 분석합니다.
지식 그래프를 구축하고 커뮤니티를 탐지합니다.

출력:
- cooccurrence_matrix.csv: 한자-현토 공기 행렬
- cooccurrence_network.html: 인터랙티브 네트워크 시각화
- cooccurrence_analysis.md: 분석 리포트
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# 주요 현토 마커 목록
MAJOR_MARKERS = [
    "라", "니라", "하니라", "시니라", "이라",
    "는", "은", "을", "를", "에", "로",
    "하니", "하여", "하고", "하야", "하다",
    "면", "니", "요", "이니", "이요",
    "잇고", "잇가", "러니", "리오", "리라",
    "되", "대", "며", "고"
]

# 유교 핵심 한자 (확장)
CORE_HANJA = [
    "子", "曰", "之", "也", "不", "而", "以", "者", "其", "為",
    "人", "德", "仁", "義", "禮", "智", "信", "忠", "孝", "道",
    "天", "地", "君", "臣", "父", "母", "民", "國", "王", "聖",
    "學", "知", "心", "性", "理", "氣", "善", "惡", "中", "和",
    "言", "行", "事", "物", "時", "日", "月", "年", "大", "小"
]


def extract_hanja(text: str) -> List[str]:
    """텍스트에서 한자 추출 (CJK 유니코드 범위)"""
    if not text or pd.isna(text):
        return []
    # CJK 통합 한자 범위
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return pattern.findall(str(text))


def extract_markers_simple(text: str) -> List[str]:
    """텍스트에서 현토 마커 추출 (단순화된 버전)"""
    if not text or pd.isna(text):
        return []
    
    found = []
    text_str = str(text)
    for marker in sorted(MAJOR_MARKERS, key=len, reverse=True):
        if marker in text_str:
            found.append(marker)
    return list(set(found))


def build_cooccurrence_matrix(
    df: pd.DataFrame,
    top_hanja: int = 100,
    top_markers: int = 30
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """한자-현토 공기 행렬 구축"""
    
    # 전체 한자/마커 빈도 계산
    hanja_counts = Counter()
    marker_counts = Counter()
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    print("  공기 패턴 수집 중...")
    for idx, row in df.iterrows():
        src_left = str(row.get("src_left", ""))
        src_right = str(row.get("src_right", ""))
        combined = src_left + " " + src_right
        
        hanja_list = extract_hanja(combined)
        marker_list = extract_markers_simple(combined)
        
        for h in hanja_list:
            hanja_counts[h] += 1
        for m in marker_list:
            marker_counts[m] += 1
        
        # 공기 카운트
        for h in set(hanja_list):
            for m in set(marker_list):
                cooccurrence[h][m] += 1
        
        if idx % 50000 == 0:
            print(f"    진행: {idx}/{len(df)}")
    
    # 상위 한자/마커 선택
    top_hanja_list = [h for h, _ in hanja_counts.most_common(top_hanja)]
    top_marker_list = [m for m, _ in marker_counts.most_common(top_markers)]
    
    # 행렬 구축
    matrix_data = []
    for h in top_hanja_list:
        row = {"hanja": h, "hanja_freq": hanja_counts[h]}
        for m in top_marker_list:
            row[m] = cooccurrence[h].get(m, 0)
        matrix_data.append(row)
    
    df_matrix = pd.DataFrame(matrix_data)
    
    return df_matrix, dict(hanja_counts), dict(marker_counts)


def compute_associations(
    df_matrix: pd.DataFrame,
    hanja_counts: Dict,
    marker_counts: Dict,
    total: int
) -> List[Dict]:
    """한자-현토 연관 강도 계산 (PMI 기반)"""
    
    associations = []
    marker_cols = [c for c in df_matrix.columns if c not in ["hanja", "hanja_freq"]]
    
    for _, row in df_matrix.iterrows():
        hanja = row["hanja"]
        h_freq = row["hanja_freq"]
        
        for marker in marker_cols:
            cooc = row[marker]
            if cooc == 0:
                continue
            
            m_freq = marker_counts.get(marker, 0)
            if m_freq == 0:
                continue
            
            # PMI (Pointwise Mutual Information)
            p_hm = cooc / total
            p_h = h_freq / total
            p_m = m_freq / total
            
            if p_h > 0 and p_m > 0 and p_hm > 0:
                pmi = np.log2(p_hm / (p_h * p_m))
                
                associations.append({
                    "hanja": hanja,
                    "marker": marker,
                    "cooccurrence": cooc,
                    "pmi": pmi,
                    "hanja_freq": h_freq,
                    "marker_freq": m_freq,
                })
    
    associations.sort(key=lambda x: x["pmi"], reverse=True)
    return associations


def generate_network_html(
    associations: List[Dict],
    out_path: Path,
    top_edges: int = 200
) -> None:
    """인터랙티브 네트워크 시각화 생성"""
    
    top_assoc = associations[:top_edges]
    
    # 노드 및 엣지 데이터 생성
    nodes = set()
    edges = []
    
    for a in top_assoc:
        nodes.add(("hanja", a["hanja"]))
        nodes.add(("marker", a["marker"]))
        edges.append({
            "source": a["hanja"],
            "target": a["marker"],
            "weight": a["pmi"],
            "cooc": a["cooccurrence"]
        })
    
    node_list = [
        {"id": n, "group": "hanja" if t == "hanja" else "marker"}
        for t, n in nodes
    ]
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>한자-현토 공기 네트워크</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 0; }}
        svg {{ width: 100%; height: 100vh; }}
        .node-hanja {{ fill: #e74c3c; }}
        .node-marker {{ fill: #3498db; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        .label {{ font-size: 12px; pointer-events: none; }}
        #info {{ position: fixed; top: 10px; left: 10px; background: white; padding: 10px; border: 1px solid #ccc; }}
    </style>
</head>
<body>
    <div id="info">
        <h3>한자-현토 공기 네트워크</h3>
        <p>노드: 빨강=한자, 파랑=현토</p>
        <p>엣지: PMI 기반 연관 강도</p>
        <p>총 {len(node_list)}개 노드, {len(edges)}개 엣지</p>
    </div>
    <svg></svg>
    <script>
        const nodes = {json.dumps(node_list, ensure_ascii=False)};
        const links = {json.dumps(edges, ensure_ascii=False)};
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const svg = d3.select("svg")
            .attr("viewBox", [0, 0, width, height]);
        
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        const link = svg.append("g")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.max(1, d.weight));
        
        const node = svg.append("g")
            .selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("r", 8)
            .attr("class", d => "node-" + d.group)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        const label = svg.append("g")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .attr("class", "label")
            .text(d => d.id);
        
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            label
                .attr("x", d => d.x + 10)
                .attr("y", d => d.y + 4);
        }});
        
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}
        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}
        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>"""
    
    out_path.write_text(html_content, encoding="utf-8")


def write_cooccurrence_report(
    out_dir: Path,
    df_matrix: pd.DataFrame,
    associations: List[Dict],
    analysis_type: str
) -> None:
    """공기 분석 리포트 작성"""
    
    lines = [
        f"# {analysis_type} 한자-현토 공기 네트워크 분석 리포트",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 1. 개요",
        "",
        f"- 분석 한자 수: {len(df_matrix)}개",
        f"- 분석 현토 수: {len([c for c in df_matrix.columns if c not in ['hanja', 'hanja_freq']])}개",
        f"- 유의미한 연관 쌍: {len(associations)}개",
        "",
        "---",
        "",
        "## 2. 상위 PMI 연관 쌍 (한자 → 현토)",
        "",
        "PMI(Pointwise Mutual Information)가 높을수록 해당 한자와 현토가 **우연 이상으로 자주** 함께 나타남을 의미합니다.",
        "",
        "| 순위 | 한자 | 현토 | 공기 횟수 | PMI |",
        "|:---:|:---:|:---:|---:|---:|",
    ]
    
    for i, a in enumerate(associations[:50], 1):
        lines.append(f"| {i} | {a['hanja']} | {a['marker']} | {a['cooccurrence']:,} | {a['pmi']:.3f} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 3. 유교 핵심 한자별 현토 선호",
        "",
    ])
    
    core_in_matrix = df_matrix[df_matrix["hanja"].isin(CORE_HANJA[:20])].copy()
    marker_cols = [c for c in df_matrix.columns if c not in ["hanja", "hanja_freq"]]
    
    for _, row in core_in_matrix.iterrows():
        hanja = row["hanja"]
        top_markers = sorted(
            [(m, row[m]) for m in marker_cols if row[m] > 0],
            key=lambda x: x[1], reverse=True
        )[:5]
        if top_markers:
            marker_str = ", ".join([f"{m}({cnt})" for m, cnt in top_markers])
            lines.append(f"- **{hanja}**: {marker_str}")
    
    lines.extend([
        "",
        "---",
        "",
        "## 4. 발견 사항",
        "",
        "### 4.1 강한 결합 패턴",
        "",
        "특정 한자는 특정 현토와 매우 강하게 결합합니다:",
        "",
    ])
    
    # 상위 10개 패턴 설명
    for a in associations[:10]:
        lines.append(f"- **{a['hanja']}** + **{a['marker']}**: PMI={a['pmi']:.3f}, 공기={a['cooccurrence']:,}회")
    
    lines.extend([
        "",
        "### 4.2 사서 특화 패턴",
        "",
        "사서(四書)에서 특징적으로 나타나는 한자-현토 조합을 별도 분석 필요.",
        "",
        "---",
        "",
        "**분석 완료**",
    ])
    
    (out_dir / f"cooccurrence_analysis_{analysis_type.lower()}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="한자-현토 공기 네트워크 분석")
    p.add_argument("--input", type=str, required=True, help="입력 CSV (boundary_clusters.csv)")
    p.add_argument("--out-dir", type=str, required=True, help="출력 디렉토리")
    p.add_argument("--analysis-type", type=str, default="SA", help="분석 타입 (PA/SA)")
    p.add_argument("--top-hanja", type=int, default=100, help="분석할 상위 한자 수")
    p.add_argument("--top-markers", type=int, default=30, help="분석할 상위 마커 수")
    p.add_argument("--top-edges", type=int, default=200, help="네트워크에 표시할 상위 엣지 수")
    
    args = p.parse_args()
    
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/4] 데이터 로딩: {in_path}")
    df = pd.read_csv(in_path, encoding="utf-8-sig")
    print(f"  -> {len(df)}건 로드")
    
    print(f"[2/4] 공기 행렬 구축...")
    df_matrix, hanja_counts, marker_counts = build_cooccurrence_matrix(
        df, top_hanja=args.top_hanja, top_markers=args.top_markers
    )
    
    # 행렬 저장
    df_matrix.to_csv(
        out_dir / f"cooccurrence_matrix_{args.analysis_type.lower()}.csv",
        index=False, encoding="utf-8-sig"
    )
    
    print(f"[3/4] 연관 강도 계산 (PMI)...")
    total = len(df)
    associations = compute_associations(df_matrix, hanja_counts, marker_counts, total)
    
    # 연관 쌍 저장
    pd.DataFrame(associations[:500]).to_csv(
        out_dir / f"associations_{args.analysis_type.lower()}.csv",
        index=False, encoding="utf-8-sig"
    )
    
    print(f"[4/4] 시각화 및 리포트 생성...")
    generate_network_html(
        associations,
        out_dir / f"cooccurrence_network_{args.analysis_type.lower()}.html",
        top_edges=args.top_edges
    )
    
    write_cooccurrence_report(out_dir, df_matrix, associations, args.analysis_type)
    
    print(f"완료: {out_dir}")
    
    # 요약 출력
    print("\n=== 요약 ===")
    print(f"분석 한자: {len(df_matrix)}개")
    print(f"유의미 연관 쌍: {len(associations)}개")
    print("상위 5개 연관:")
    for a in associations[:5]:
        print(f"  {a['hanja']} + {a['marker']}: PMI={a['pmi']:.3f}")


if __name__ == "__main__":
    main()
