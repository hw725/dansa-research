"""Extract 3-model consensus example candidates for each section x 4 categories."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODELS = ["gpt5mini", "gemini", "claude_sonnet"]
KEY = ["book", "문단식별자", "문장식별자", "marker_type"]
N_CANDIDATES = 7

SECTIONS = {
    "section1": {
        "csv": "section1_judgments.csv",
        "supplement": "supplement_section1_judgments.csv",
        "target_mt": "로다",
        "control_mt": "라(대조군)",
    },
    "section2": {
        "csv": "section2_decision_judgments.csv",
        "supplement": "supplement_section2_judgments.csv",
        "target_mt": "니라",
        "control_mt": "라",
    },
    "section3": {
        "csv": "section3_judgments.csv",
        "supplement": "supplement_section3_judgments.csv",
        "target_mt": "하나니라",
        "control_mt": "라(대조군)",
    },
}

TEXT_COLS = ["book", "문단식별자", "문장식별자", "원문", "번역문",
            "marker_raw", "marker_normalized", "dansa_category"]

all_results = {}

for section_name, cfg in SECTIONS.items():
    print(f"\n{'='*60}")
    print(f"  {section_name}: {cfg['csv']}")
    print(f"{'='*60}")

    frames = {}
    first_df = None
    for m in MODELS:
        paths = [RESULTS / m / cfg["csv"]]
        if cfg.get("supplement"):
            paths.append(RESULTS / m / cfg["supplement"])
        parts = [pd.read_csv(p, encoding="utf-8", on_bad_lines="skip") for p in paths if p.exists()]
        df = pd.concat(parts, ignore_index=True)
        sub = df[KEY + ["llm_judgment"]].drop_duplicates(subset=KEY)
        sub = sub.rename(columns={"llm_judgment": f"j_{m}"})
        frames[m] = sub
        if first_df is None:
            first_df = df

    merged = frames["gpt5mini"]
    for m in ["gemini", "claude_sonnet"]:
        merged = merged.merge(frames[m], on=KEY, how="inner")

    extra_cols = [c for c in TEXT_COLS if c not in KEY]
    text_frame = first_df[KEY + extra_cols].drop_duplicates(subset=KEY)
    merged = merged.merge(text_frame, on=KEY, how="left")

    j_cols = [c for c in merged.columns if c.startswith("j_")]
    merged["n_true"] = merged[j_cols].sum(axis=1)
    merged["consensus"] = merged["n_true"].map(
        {3: "O_agree", 0: "X_agree"}
    ).fillna("split")

    section_results = {}
    for mt, label in [(cfg["target_mt"], "target"), (cfg["control_mt"], "control")]:
        for verdict, v_label in [("O_agree", "O"), ("X_agree", "X")]:
            cat_key = f"{label}_{v_label}"
            sub = merged[(merged["marker_type"] == mt) & (merged["consensus"] == verdict)]
            n_total = len(sub)

            books = sub["book"].unique()
            samples_df = pd.DataFrame()
            if n_total > 0:
                parts = []
                for book in books:
                    bk = sub[sub["book"] == book]
                    if len(bk) > 0:
                        parts.append(bk.sample(min(2, len(bk)), random_state=42))
                    if sum(len(s) for s in parts) >= N_CANDIDATES:
                        break
                if parts:
                    samples_df = pd.concat(parts).head(N_CANDIDATES)

            print(f"\n  [{cat_key}] pool={n_total}, showing {len(samples_df)}")
            candidates = []
            for _, row in samples_df.iterrows():
                entry = {
                    "book": row["book"],
                    "문단식별자": int(row["문단식별자"]),
                    "문장식별자": int(row["문장식별자"]),
                    "원문": row["원문"],
                    "번역문": row["번역문"],
                    "marker_raw": row.get("marker_raw", ""),
                    "dansa_category": row.get("dansa_category", ""),
                }
                print(f"    {row['book']} {int(row['문단식별자'])}:{int(row['문장식별자'])}")
                print(f"      원문: {str(row['원문'])[:80]}")
                print(f"      번역: {str(row['번역문'])[:80]}")
                candidates.append(entry)
            section_results[cat_key] = {
                "pool_size": n_total,
                "candidates": candidates,
            }
    all_results[section_name] = section_results

out_path = RESULTS / "example_candidates.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_path}")
