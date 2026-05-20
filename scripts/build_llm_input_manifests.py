#!/usr/bin/env python3
"""Build local LLM input manifests from the current canonical result CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SOURCE_MODEL = "gpt5mini"
SOURCE_DIR = REPO / "results" / SOURCE_MODEL
OUT_DIR = REPO / "data" / "llm_manifests"

BASE_FILES = {
    "section1_base.csv": "section1_judgments.csv",
    "section2_base.csv": "section2_decision_judgments.csv",
    "section3_base.csv": "section3_judgments.csv",
}

DROP_COLUMNS = ["llm_judgment"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for out_name, src_name in BASE_FILES.items():
        src = SOURCE_DIR / src_name
        if not src.exists():
            raise FileNotFoundError(src)
        df = pd.read_csv(src)
        df = df.drop(columns=DROP_COLUMNS, errors="ignore")
        out = OUT_DIR / out_name
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"wrote {out.relative_to(REPO)} ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
