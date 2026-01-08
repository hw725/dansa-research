import pandas as pd
import regex as re
from collections import Counter
from pathlib import Path

# 현재 스크립트의 정규화 로직 복제
HYEONTO_REPLACE_MAP = {
    "은": "는",
    "이": "가",
    "을": "를",
    "과": "와",
    "ㅣ": "가",
}

def normalize_marker(marker: str) -> str:
    if not marker: return marker
    if marker in HYEONTO_REPLACE_MAP:
        return HYEONTO_REPLACE_MAP[marker]
    if len(marker) > 1:
        if marker.startswith("이") or marker.startswith("으"):
            return marker[1:]
    return marker

_CJK_MARKER_RE = re.compile(r"(?P<cjk>\p{Han}+)(?P<marker>\p{Hangul}+)?")

def extract_markers(text):
    if pd.isna(text): return []
    out = []
    for m in _CJK_MARKER_RE.finditer(str(text)):
        marker = m.group("marker")
        if marker: out.append(marker)
    return out

def main():
    csv_path = Path("hyeonto/reports/recluster_k16_child_minper50/reclustered.csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    src_cols = ["src_left", "src_right"]
    
    all_markers = []
    for col in src_cols:
        if col in df.columns:
            for val in df[col]:
                all_markers.extend(extract_markers(val))
    
    counter = Counter(all_markers)
    
    print(f"{'Raw Marker':<15} | {'Count':<8} | {'Normalized':<15}")
    print("-" * 45)
    
    # 빈도순 상위 100개 출력
    for marker, count in counter.most_common(100):
        norm = normalize_marker(marker)
        print(f"{marker:<15} | {count:<8} | {norm:<15}")

if __name__ == "__main__":
    main()
