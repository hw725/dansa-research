import json
from pathlib import Path
from typing import List, Dict
import pandas as pd

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets"
OUT_ROOT = Path(__file__).resolve().parents[1] / "datasets"


def make_samples_from_csv(csv_path: Path, field: str, window_size: int) -> List[Dict]:
    df = pd.read_csv(csv_path, dtype=str)
    for col in df.columns:
        df[col] = df[col].fillna("")
    # 정렬: series, book, row_index
    if "row_index" in df.columns:
        df["row_index"] = df["row_index"].astype(int)
    df = df.sort_values(by=["series", "book", "row_index"]) if "series" in df.columns else df

    samples: List[Dict] = []
    # 그룹핑: 같은 책 단위
    for (series, book), g in df.groupby(["series", "book"]):
        rows = g[field].tolist()
        # 윈도우로 묶기
        for i in range(0, len(rows), window_size):
            chunk = rows[i:i+window_size]
            text = "".join(chunk)
            # 경계 라벨: 각 문장/문단의 마지막 문자 위치에 B, 나머지 O
            labels = ["O"] * len(text)
            offset = 0
            for seg in chunk:
                if len(seg) == 0:
                    continue
                end_idx = offset + len(seg) - 1
                if 0 <= end_idx < len(labels):
                    labels[end_idx] = "B"
                offset += len(seg)
            samples.append({
                "series": series,
                "book": book,
                "text": text,
                "labels": "".join(labels),
                "field": field,
                "window_size": window_size,
            })
    return samples


def build_boundary_dataset(src_ds: str, field: str, window_size: int = 20):
    src_root = DATASETS_ROOT / src_ds
    # src/tgt별 별도 디렉토리
    suffix = "_src_boundary" if field == "src" else "_boundary"
    out_root = OUT_ROOT / f"{src_ds}{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        csv_path = src_root / f"{split}.csv"
        if not csv_path.exists():
            continue
        samples = make_samples_from_csv(csv_path, field=field, window_size=window_size)
        out_path = out_root / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{src_ds} ({field}) → {split}: {len(samples)} samples → {out_path}")


def main():
    # pa: 문장 병렬 (한국어 번역문 tgt 기준)
    build_boundary_dataset("pa", field="tgt", window_size=20)
    # pd: 문단 병렬 (한국어 번역문 tgt 기준)
    build_boundary_dataset("pd", field="tgt", window_size=10)
    # sa: 구병렬 (한국어 번역문 tgt 기준)
    build_boundary_dataset("sa", field="tgt", window_size=20)

    # 원문(src) 기준 경계 태깅도 병행 생성
    build_boundary_dataset("pa", field="src", window_size=20)
    build_boundary_dataset("pd", field="src", window_size=10)
    build_boundary_dataset("sa", field="src", window_size=20)
    print("✅ B/O 경계 태깅용 JSONL 생성 완료")


if __name__ == "__main__":
    main()
