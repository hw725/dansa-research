import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import regex as re

def extract_markers(text):
    if pd.isna(text): return []
    # Simplified marker extraction for trend analysis
    return re.findall(r"[\p{Hangul}]+", str(text))

def main():
    csv_path = Path("hyeonto/reports/recluster_k16_child/reclustered.csv")
    if not csv_path.exists():
        print("CSV not found.")
        return

    df = pd.read_csv(csv_path)
    SASEO_BOOKS = ["논어", "맹자", "대학", "중용"]
    
    # 1. Cluster Stats
    def clean_pid(x):
        s = str(x)
        if s.startswith('p'): return int(s[1:])
        return int(x)
        
    df["pid"] = df["parent_cluster_id"].apply(clean_pid)
    
    stats = []
    for pid in sorted(df["pid"].unique()):
        pdf = df[df["pid"] == pid]
        total_rows = len(pdf)
        
        # Canonicity
        saseo_mask = pdf["book_name"].str.contains("|".join(SASEO_BOOKS), na=False)
        canonicity = (len(pdf[saseo_mask]) / total_rows) * 100
        
        # Marker Stats
        markers = []
        for col in ["src_left", "src_right"]:
            if col in pdf.columns:
                for val in pdf[col]:
                    markers.extend(extract_markers(val))
        
        m_counts = Counter(markers)
        unique_m = len(m_counts)
        total_m = sum(m_counts.values())
        diversity = (unique_m / total_m) * 100 if total_m > 0 else 0
        
        # Specific Marker Ratios
        # '라' (Final/Definitive) vs '하니' (Connective/Narrative)
        ratio_ra = (m_counts.get("라", 0) / total_m) * 100 if total_m > 0 else 0
        ratio_hani = (m_counts.get("하니", 0) / total_m) * 100 if total_m > 0 else 0
        
        stats.append({
            "pid": pid,
            "size": total_rows,
            "canonicity": canonicity,
            "diversity": diversity,
            "ra_ratio": ratio_ra,
            "hani_ratio": ratio_hani
        })
        
    res_df = pd.DataFrame(stats)
    print("\n### Correlation Analysis")
    print(res_df.corr()[["canonicity"]].drop("canonicity"))
    
    print("\n### Cluster Table for Trend Analysis")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
