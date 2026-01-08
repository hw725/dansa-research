import pandas as pd
import regex as re
from collections import Counter, defaultdict
from pathlib import Path

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
    # 최신 전체 데이터셋 경로로 업데이트
    csv_path = Path("hyeonto/reports/recluster_k16_child/reclustered.csv")
    if not csv_path.exists():
        # Fallback to old path if not found
        csv_path = Path("hyeonto/reports/recluster_k16_child_minper50/reclustered.csv")
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    print(f"Generating profile for: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # parent_cluster_id가 'p0' 형태이거나 0(int) 형태인 경우 대응
    def clean_pid(x):
        s = str(x)
        if s.startswith('p'): return s[1:]
        return s
        
    df["parent_id"] = df["parent_cluster_id"].apply(clean_pid)
    
    # 조선 공인 정본 (사서만 Canon으로 인정)
    CANON_BOOKS = ["논어", "맹자", "대학", "중용"]
    
    # 세분화된 가중치 설계
    WEIGHT_MAP = {
        "논어": 5.0, "맹자": 5.0, "대학": 5.0, "중용": 5.0,
        "서경": 3.0,
        "시경": 2.0, "주역": 2.0
    }
    
    report = []
    parent_ids = sorted(df["parent_id"].unique(), key=lambda x: int(x))

    for pid in parent_ids:
        pdf = df[df["parent_id"] == pid]
        total_rows = len(pdf)
        
        # 1. Source Distribution (서종별 분포)
        book_counts = pdf["book_name"].value_counts()
        top_books = book_counts.head(4)
        book_dist_str = ", ".join([f"{b}({v/total_rows*100:.1f}%)" for b, v in top_books.items()])
        
        # 2. Canonicity (Canon Ratio - 사서 기준)
        canon_mask = pdf["book_name"].str.contains("|".join(CANON_BOOKS), na=False)
        canon_rows = len(pdf[canon_mask])
        canonicity = (canon_rows / total_rows) * 100
        
        # 3. Weighted Markers (차등 가중치 적용)
        marker_scores = defaultdict(float)
        for _, row in pdf.iterrows():
            book_name = str(row.get("book_name", ""))
            
            # 가중치 결정 로직
            weight = 1.0
            for kw, w in WEIGHT_MAP.items():
                if kw in book_name:
                    weight = w
                    break
            
            for col in ["src_left", "src_right"]:
                if col in pdf.columns:
                    mks = extract_markers(row[col])
                    for m in mks:
                        marker_scores[m] += weight
        
        top_markers = sorted(marker_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # 4. Strategic Examples (사서 우선)
        canon_pdf = pdf[canon_mask]
        other_pdf = pdf[~canon_mask]
        examples = []
        for _, row in canon_pdf.sample(min(len(canon_pdf), 7), random_state=42).iterrows():
            examples.append(f"⭐ [사서] [{row.get('book_name')}] {row.get('src_left', '')} | {row.get('src_right', '')}")
        
        if len(examples) < 10:
            remaining = 10 - len(examples)
            for _, row in other_pdf.sample(min(len(other_pdf), remaining), random_state=42).iterrows():
                examples.append(f"  [{row.get('book_name')}] {row.get('src_left', '')} | {row.get('src_right', '')}")

        report.append({
            "pid": pid,
            "count": total_rows,
            "canonicity": canonicity,
            "book_dist": book_dist_str,
            "top_markers": ", ".join([f"{m}({int(s)})" for m, s in top_markers]),
            "examples": examples
        })

    # Save report
    with open("hyeonto/reports/parent_cluster_profile.md", "w", encoding="utf-8") as f:
        f.write("# Parent Cluster Profile Analysis (Tiered Authority Analysis)\n\n")
        f.write("이 보고서는 도서별 차등 가중치(**사서 5x, 서경 3x, 시경/주역 2x**)를 반영한 현토 분석 결과입니다.\n\n")
        f.write("- **Dominant Sources**: 클러스터 내 주요 문헌 분포입니다.\n")
        f.write("- **Canonicity (Saseo)**: 논어·맹자·대학·중용 정본이 차지하는 비중입니다.\n")
        f.write("- **Weighted Markers**: 전적별 위계적 권위를 반영하여 도출한 핵심 현토입니다.\n\n")
        f.write("---\n\n")

        for item in report:
            f.write(f"## Cluster p{item['pid']} (Size: {item['count']} rows)\n")
            f.write(f"- **Dominant Sources**: {item['book_dist']}\n")
            f.write(f"- **Canonicity (Saseo Ratio)**: {item['canonicity']:.2f}%\n")
            f.write(f"- **Top Markers (Weighted)**: {item['top_markers']}\n\n")
            f.write("- **Example Contexts**:\n")
            for ex in item["examples"]:
                f.write(f"  - {ex}\n")
            f.write("\n---\n\n")
    
    print("Report generated: hyeonto/reports/parent_cluster_profile.md")

if __name__ == "__main__":
    main()
