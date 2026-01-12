#!/usr/bin/env python3
"""
PA와 SA 클러스터링 결과 비교 분석 (시각화 포함)

PA: 문장↔문장 경계 (87K 경계)
SA: 구↔구 경계 (295K 경계)

비교 내용:
1. 클러스터 크기 분포
2. 서종별 분포
3. 현토 분포 비교
4. 시각화 (Plotly)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import regex as re

# 현토 추출
_CJK_MARKER_RE = re.compile(r"(?P<cjk>\p{Han}+)(?P<marker>\p{Hangul}+)?")
HYEONTO_REPLACE_MAP = {"은": "는", "이": "가", "을": "를", "과": "와", "ㅣ": "가"}

def normalize_marker(marker: str) -> str:
    if not marker:
        return marker
    if marker in HYEONTO_REPLACE_MAP:
        return HYEONTO_REPLACE_MAP[marker]
    if len(marker) > 1 and marker[0] in ("이", "으"):
        return marker[1:]
    return marker

def extract_markers(text: object) -> list[str]:
    if text is None or str(text) == "nan":
        return []
    out = []
    for m in _CJK_MARKER_RE.finditer(str(text)):
        marker = m.group("marker")
        if marker:
            out.append(normalize_marker(marker))
    return out

def analyze_cluster_data(df: pd.DataFrame, src_cols: list[str], label: str) -> dict:
    """클러스터 데이터 분석"""
    stats = {
        "label": label,
        "total_rows": len(df),
        "num_clusters": df["cluster_id"].nunique(),
        "cluster_sizes": df["cluster_id"].value_counts().to_dict(),
    }
    
    # 사서 비중
    CANON_BOOKS = ["논어", "맹자", "대학", "중용"]
    if "book_name" in df.columns:
        canon_mask = df["book_name"].str.contains("|".join(CANON_BOOKS), na=False)
        stats["canon_ratio"] = len(df[canon_mask]) / len(df) * 100
        stats["book_dist"] = df["book_name"].value_counts().head(10).to_dict()
        stats["num_books"] = df["book_name"].nunique()
    
    # 현토 빈도
    marker_counts = defaultdict(int)
    for _, row in df.iterrows():
        for col in src_cols:
            if col in df.columns:
                for m in extract_markers(row.get(col)):
                    marker_counts[m] += 1
    
    stats["top_markers"] = dict(sorted(marker_counts.items(), key=lambda x: -x[1])[:30])
    stats["total_markers"] = sum(marker_counts.values())
    stats["unique_markers"] = len(marker_counts)
    
    return stats

def generate_comparison_html(pa_stats: dict, sa_stats: dict, out_path: Path):
    """비교 시각화 HTML 생성"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly 미설치 - HTML 생성 생략")
        return
    
    # 현토 상위 20개 비교 바차트
    pa_markers = list(pa_stats["top_markers"].keys())[:20]
    sa_markers = list(sa_stats["top_markers"].keys())[:20]
    all_markers = list(dict.fromkeys(pa_markers + sa_markers))[:25]
    
    pa_counts = [pa_stats["top_markers"].get(m, 0) for m in all_markers]
    sa_counts = [sa_stats["top_markers"].get(m, 0) for m in all_markers]
    
    # 정규화 (비율로 변환)
    pa_total = pa_stats["total_markers"] or 1
    sa_total = sa_stats["total_markers"] or 1
    pa_ratios = [c / pa_total * 100 for c in pa_counts]
    sa_ratios = [c / sa_total * 100 for c in sa_counts]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "현토 분포 비교 (상위 25개, 비율 %)",
            "클러스터 크기 분포",
            "데이터셋 개요",
            "도서 수 비교"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "table"}, {"type": "bar"}]
        ]
    )
    
    # 1. 현토 분포 비교
    fig.add_trace(
        go.Bar(name=f"PA ({pa_stats['total_rows']:,})", x=all_markers, y=pa_ratios, marker_color="steelblue"),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name=f"SA ({sa_stats['total_rows']:,})", x=all_markers, y=sa_ratios, marker_color="coral"),
        row=1, col=1
    )
    
    # 2. 클러스터 크기 분포
    pa_sizes = sorted(pa_stats["cluster_sizes"].values(), reverse=True)
    sa_sizes = sorted(sa_stats["cluster_sizes"].values(), reverse=True)
    fig.add_trace(
        go.Bar(name="PA Clusters", x=list(range(len(pa_sizes))), y=pa_sizes, marker_color="steelblue", showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(name="SA Clusters", x=list(range(len(sa_sizes))), y=sa_sizes, marker_color="coral", showlegend=False),
        row=1, col=2
    )
    
    # 3. 개요 테이블
    fig.add_trace(
        go.Table(
            header=dict(values=["지표", "PA (문장)", "SA (구)"], fill_color="lightgray", font=dict(size=14)),
            cells=dict(values=[
                ["총 경계 수", "클러스터 수", "도서 수", "고유 현토 수", "사서 비중"],
                [f"{pa_stats['total_rows']:,}", pa_stats['num_clusters'], pa_stats.get('num_books', 'N/A'), pa_stats['unique_markers'], f"{pa_stats.get('canon_ratio', 0):.1f}%"],
                [f"{sa_stats['total_rows']:,}", sa_stats['num_clusters'], sa_stats.get('num_books', 'N/A'), sa_stats['unique_markers'], f"{sa_stats.get('canon_ratio', 0):.1f}%"]
            ], font=dict(size=12))
        ),
        row=2, col=1
    )
    
    # 4. 도서 수 비교
    fig.add_trace(
        go.Bar(
            x=["PA", "SA"],
            y=[pa_stats.get('num_books', 0), sa_stats.get('num_books', 0)],
            marker_color=["steelblue", "coral"],
            text=[pa_stats.get('num_books', 0), sa_stats.get('num_books', 0)],
            textposition="outside"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="PA vs SA 클러스터 비교 분석 (v2 통합 데이터)",
        height=900,
        width=1400,
        barmode="group",
        font=dict(family="NanumGothic, Arial", size=12)
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"HTML 저장: {out_path}")

def generate_markdown_report(pa_stats: dict, sa_stats: dict, out_path: Path):
    """마크다운 보고서 생성"""
    md_lines = [
        "# PA vs SA 클러스터 비교 분석",
        "",
        f"**분석 일시**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 개요",
        "",
        "| 지표 | PA (문장 경계) | SA (구 경계) | 비고 |",
        "|:---|---:|---:|:---|",
        f"| 총 경계 수 | {pa_stats['total_rows']:,} | {sa_stats['total_rows']:,} | SA가 {sa_stats['total_rows']/pa_stats['total_rows']:.1f}배 |",
        f"| 클러스터 수 | {pa_stats['num_clusters']} | {sa_stats['num_clusters']} | - |",
        f"| 도서 수 | {pa_stats.get('num_books', 'N/A')} | {sa_stats.get('num_books', 'N/A')} | - |",
        f"| 고유 현토 수 | {pa_stats['unique_markers']} | {sa_stats['unique_markers']} | - |",
        f"| 사서 비중 | {pa_stats.get('canon_ratio', 0):.1f}% | {sa_stats.get('canon_ratio', 0):.1f}% | - |",
        "",
        "## 2. 현토 분포 Top 15",
        "",
        "| 순위 | PA 현토 | PA 빈도 | SA 현토 | SA 빈도 |",
        "|:---:|:---:|---:|:---:|---:|",
    ]
    
    pa_top = list(pa_stats["top_markers"].items())[:15]
    sa_top = list(sa_stats["top_markers"].items())[:15]
    for i in range(15):
        pa_m, pa_c = pa_top[i] if i < len(pa_top) else ("", 0)
        sa_m, sa_c = sa_top[i] if i < len(sa_top) else ("", 0)
        md_lines.append(f"| {i+1} | {pa_m} | {pa_c:,} | {sa_m} | {sa_c:,} |")
    
    md_lines.extend([
        "",
        "## 3. 해석",
        "",
        "- **PA (문장 경계)**: 문장 간 경계에서의 현토 사용 패턴을 분석. 장거리 문맥 의존성 포착에 유리.",
        "- **SA (구 경계)**: 구 단위의 국소적 현토 패턴을 분석. 번역문과의 직접 대응 분석에 유리.",
        "- **SA가 PA보다 약 3~4배 많은 이유**: 하나의 문장 내에 여러 구가 존재하기 때문.",
        "",
        "## 4. 시각화",
        "",
        "- `pa_sa_comparison.html`: 인터랙티브 비교 차트",
        "",
        "---",
        f"*자동 생성: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*"
    ])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"마크다운 저장: {out_path}")

def main():
    p = argparse.ArgumentParser(description="PA vs SA 클러스터 비교 분석")
    p.add_argument("--pa-csv", type=Path, default=Path("hyeonto/reports/boundary_function_clusters/boundary_clusters.csv"))
    p.add_argument("--sa-csv", type=Path, default=Path("hyeonto/reports/sa_boundary_clusters/sa_boundary_clusters.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("hyeonto/reports/pa_sa_comparison"))
    p.add_argument("--src-cols", type=str, default="src_left,src_right")
    args = p.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    src_cols = [c.strip() for c in args.src_cols.split(",")]
    
    print("=== PA 클러스터 로드 ===")
    pa = pd.read_csv(args.pa_csv)
    pa_stats = analyze_cluster_data(pa, src_cols, "PA (문장 경계)")
    print(f"PA: {pa_stats['total_rows']:,}개 경계, {pa_stats['num_clusters']}개 클러스터")
    
    print("\n=== SA 클러스터 로드 ===")
    sa = pd.read_csv(args.sa_csv)
    sa_stats = analyze_cluster_data(sa, src_cols, "SA (구 경계)")
    print(f"SA: {sa_stats['total_rows']:,}개 경계, {sa_stats['num_clusters']}개 클러스터")
    
    # JSON 저장
    with open(args.out_dir / "pa_stats.json", "w", encoding="utf-8") as f:
        json.dump(pa_stats, f, ensure_ascii=False, indent=2, default=str)
    with open(args.out_dir / "sa_stats.json", "w", encoding="utf-8") as f:
        json.dump(sa_stats, f, ensure_ascii=False, indent=2, default=str)
    
    # HTML 시각화
    generate_comparison_html(pa_stats, sa_stats, args.out_dir / "pa_sa_comparison.html")
    
    # 마크다운 보고서
    generate_markdown_report(pa_stats, sa_stats, args.out_dir / "comparison_report.md")
    
    # 요약 JSON
    summary = {
        "pa_boundaries": pa_stats["total_rows"],
        "pa_clusters": pa_stats["num_clusters"],
        "pa_books": pa_stats.get("num_books", 0),
        "sa_boundaries": sa_stats["total_rows"],
        "sa_clusters": sa_stats["num_clusters"],
        "sa_books": sa_stats.get("num_books", 0),
        "ratio_sa_to_pa": sa_stats["total_rows"] / pa_stats["total_rows"] if pa_stats["total_rows"] > 0 else 0,
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n완료: {args.out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

