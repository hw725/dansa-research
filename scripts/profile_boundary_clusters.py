import pandas as pd
import regex as re
from collections import Counter, defaultdict
from pathlib import Path
import json

# Reuse normalization logic
HYEONTO_REPLACE_MAP = { "은": "는", "이": "가", "을": "를", "과": "와", "ㅣ": "가" }
def normalize_marker(marker: str) -> str:
    if not marker: return marker
    if marker in HYEONTO_REPLACE_MAP: return HYEONTO_REPLACE_MAP[marker]
    if len(marker) > 1 and (marker.startswith("이") or marker.startswith("으")): return marker[1:]
    return marker

_CJK_MARKER_RE = re.compile(r"(?P<cjk>\p{Han}+)(?P<marker>\p{Hangul}+)?")
def extract_markers(text):
    if pd.isna(text): return []
    return [normalize_marker(m.group("marker")) for m in _CJK_MARKER_RE.finditer(str(text)) if m.group("marker")]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, help="Input cluster CSV path")
    parser.add_argument("--out", type=Path, help="Output report path")
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path:
        csv_path = Path("hyeonto/reports/phrase_boundary_clusters/sa_boundary_clusters.csv")
        if not csv_path.exists():
            csv_path = Path("hyeonto/reports/boundary_function_clusters/boundary_clusters.csv")
    
    if not csv_path or not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    out_path = args.out
    if not out_path:
        out_path = Path("hyeonto/reports/parent_cluster_profile.md")
    
    print(f"Generating profile for: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1인칭/겸양 지표 한자
    HUMBLE_HANJA = ['吾', '我', '予', '朕', '臣', '竊', '伏', '下', '卑', '愚']
    # 의문사 지표 한자
    INTERROGATIVE_HANJA = ['何', '誰', '孰', '豈', '焉', '焉', '焉', '胡', '曷', '安']
    
    # 도서별 가중치 (Tiered Authority)
    WEIGHT_MAP = {
        "논어": 5.0, "맹자": 5.0, "대학": 5.0, "중용": 5.0,
        "서경": 3.0,
        "시경": 2.0, "주역": 2.0
    }
    CANON_BOOKS = ["논어", "맹자", "대학", "중용"]
    
    report = []
    cluster_ids = sorted(df["cluster_id"].unique())

    for cid in cluster_ids:
        cdf = df[df["cluster_id"] == cid]
        total_rows = len(cdf)
        
        # 1. Source Distribution
        book_counts = cdf["book_name"].value_counts()
        top_books = book_counts.head(5)
        book_dist_str = ", ".join([f"{b}({v/total_rows*100:.1f}%)" for b, v in top_books.items()])
        
        # 2. Canonicity (Saseo Ratio)
        canon_mask = cdf["book_name"].str.contains("|".join(CANON_BOOKS), na=False)
        canon_rows = len(cdf[canon_mask])
        canonicity = (canon_rows / total_rows) * 100
        
        # 3. Weighted Markers & Hanja
        marker_scores = defaultdict(float)
        hanja_scores = defaultdict(float)
        humble_score = 0
        interrogative_score = 0
        
        for _, row in cdf.iterrows():
            book_name = str(row.get("book_name", ""))
            weight = 1.0
            for kw, w in WEIGHT_MAP.items():
                if kw in book_name:
                    weight = w
                    break
            
            # Left tail & Right head focused
            for text in [row.get("src_left", ""), row.get("src_right", "")]:
                if pd.isna(text): continue
                mks = extract_markers(text)
                for m in mks:
                    marker_scores[m] += weight
                
                # Extract Hanja
                hanjas = re.findall(r'\p{Han}', str(text))
                for h in hanjas:
                    hanja_scores[h] += weight
                    if h in HUMBLE_HANJA: humble_score += weight
                    if h in INTERROGATIVE_HANJA: interrogative_score += weight
        
        top_markers = sorted(marker_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        top_hanja = sorted(hanja_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # 4. Strategic Examples
        canon_cdf = cdf[canon_mask]
        other_cdf = cdf[~canon_mask]
        examples = []
        if len(canon_cdf) > 0:
            for _, row in canon_cdf.sample(min(len(canon_cdf), 5), random_state=42).iterrows():
                examples.append(f"⭐ [사서] [{row.get('book_name')}] {row.get('src_left', '')} | {row.get('src_right', '')}")
        
        remaining = 10 - len(examples)
        if remaining > 0 and len(other_cdf) > 0:
            for _, row in other_cdf.sample(min(len(other_cdf), remaining), random_state=42).iterrows():
                examples.append(f"  [{row.get('book_name')}] {row.get('src_left', '')} | {row.get('src_right', '')}")

        report.append({
            "cid": cid,
            "count": total_rows,
            "canonicity": canonicity,
            "book_dist": book_dist_str,
            "top_markers": ", ".join([f"{m}({int(s)})" for m, s in top_markers]),
            "top_hanja": ", ".join([f"{h}" for h, s in top_hanja]),
            "humble_rank": humble_score / total_rows,
            "interrogative_rank": interrogative_score / total_rows,
            "examples": examples
        })

    # Save report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 📂 전체 클러스터 상세 프로파일 (Unified Master Profile)\n\n")
        f.write("- **데이터**: hyeonto/datasets/sentence_merged_v2.csv (통합본)\n")
        f.write("- **분석 레벨**: 문장 경계(Boundary) 비지도 군집화 (K=16)\n")
        f.write("- **가중치**: 사서(5x), 서경(3x), 시경/주역(2x) 반영\n\n")
        
        f.write("##  Summary Table\n\n")
        f.write("| ID | Size | Canonicity | Dominant Features | Humble/Intr |\n")
        f.write("|:---|:---:|:---:|:---|:---:|\n")
        for item in report:
            features = item['top_markers'].split(', ')[0]
            hi_tag = ""
            if item['humble_rank'] > 0.5: hi_tag += "🙇"
            if item['interrogative_rank'] > 0.5: hi_tag += "❓"
            f.write(f"| p{item['cid']} | {item['count']} | {item['canonicity']:.1f}% | {features} | {hi_tag} |\n")
        f.write("\n---\n\n")

        for item in report:
            f.write(f"## Cluster p{item['cid']} (Size: {item['count']} rows)\n")
            f.write(f"- **Dominant Sources**: {item['book_dist']}\n")
            f.write(f"- **Canonicity (Saseo Ratio)**: {item['canonicity']:.2f}%\n")
            f.write(f"- **Top Markers (Weighted)**: {item['top_markers']}\n")
            f.write(f"- **Core Hanja**: {item['top_hanja']}\n")
            f.write(f"- **Pragmatic Index**: Humble({item['humble_rank']:.2f}), Interrogative({item['interrogative_rank']:.2f})\n\n")
            f.write("- **Example Contexts**:\n")
            for ex in item["examples"]:
                f.write(f"  - {ex}\n")
            f.write("\n---\n\n")
    
    print(f"Report generated: {out_path}")

if __name__ == "__main__":
    main()
