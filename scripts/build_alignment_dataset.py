from pathlib import Path
import random
import json
import pandas as pd
from typing import Dict, List

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
OUT_ROOT_BASE = Path(__file__).resolve().parents[1] / "datasets" / "alignment"
random.seed(42)


def read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    for col in df.columns:
        df[col] = df[col].fillna("")
    if "row_index" in df.columns:
        df["row_index"] = df["row_index"].astype(int)
    return df


def build_split(dataset: str, split: str, max_negatives_per_pos: int = 1):
    """dataset(sa|pa|sa_gu)의 CSV로부터 alignment jsonl 생성."""
    src_root = DATASETS_ROOT / dataset
    csv_path = src_root / f"{split}.csv"
    if not csv_path.exists():
        print(f"⚠️ CSV not found: {csv_path}")
        return

    df = read_csv(csv_path)
    # 그룹: 동일 책(book_name) 내에서 음성 샘플을 뽑아 난이도 유지
    # 'series' 또는 'book_name' 컬럼 지원
    group_col = "book_name" if "book_name" in df.columns else "series"
    groups: Dict[str, pd.DataFrame] = {book: g for book, g in df.groupby(group_col)}
    
    # src/tgt 컬럼명 감지
    src_col = "원문" if "원문" in df.columns else "src"
    tgt_col = "번역문" if "번역문" in df.columns else "tgt"

    out_dir = OUT_ROOT_BASE / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.jsonl"

    total_pos = 0
    total_neg = 0

    with out_path.open("w", encoding="utf-8") as fw:
        # 책별로 한 번에 처리해 샘플링 비용 절감
        for book_name, g in groups.items():
            if g.empty:
                continue
            # 유의미 행만 필터
            g = g[(g[src_col] != "") & (g[tgt_col] != "")]
            if len(g) == 0:
                continue
            # positives: 그대로 기록
            for _, row in g.iterrows():
                item = {
                    "book": book_name,
                    "src": row.get(src_col, ""),
                    "tgt": row.get(tgt_col, ""),
                    "label": 1,
                }
                fw.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_pos += 1

            # negatives: tgt를 순환 시프트하여 src와 불일치 생성 (O(n))
            tgt_list = g[tgt_col].tolist()
            src_list = g[src_col].tolist()
            if len(tgt_list) >= 2:
                # rotate by 1
                rotated_tgt = tgt_list[1:] + tgt_list[:1]
                for src, tgt_neg in zip(src_list, rotated_tgt):
                    if src and tgt_neg:
                        item = {
                            "book": book_name,
                            "src": src,
                            "tgt": tgt_neg,
                            "label": 0,
                        }
                        fw.write(json.dumps(item, ensure_ascii=False) + "\n")
                        total_neg += 1

    print(f"{split}: pos={total_pos} neg={total_neg} → {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build alignment dataset from CSV")
    parser.add_argument("--dataset", "-d", default="sa", choices=["pa", "sa", "pd"], help="source dataset")
    parser.add_argument("--negatives", "-n", type=int, default=1, help="negatives per positive (rotation-based)")
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        build_split(args.dataset, split, max_negatives_per_pos=args.negatives)
    print(f"✅ 의미 대응 학습용 JSONL 생성 완료 ({args.dataset})")


if __name__ == "__main__":
    main()
