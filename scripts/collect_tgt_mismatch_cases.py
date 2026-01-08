import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "")
    text = text.strip()
    text = WS_RE.sub(" ", text)
    return text


def strip_all_ws(text: str) -> str:
    return WS_RE.sub("", normalize_text(text))


@dataclass(frozen=True)
class MismatchCase:
    seed: int
    book_name: str
    paragraph_id: int
    category: str
    pred_sents: list[str]
    gold_sents: list[str]


def classify(pred_sents: list[str], gold_sents: list[str]) -> str:
    if pred_sents == gold_sents:
        return "exact_equal"

    pred_norm = [normalize_text(s) for s in pred_sents]
    gold_norm = [normalize_text(s) for s in gold_sents]

    if pred_norm == gold_norm:
        return "normalization_only"

    # 문장 경계 차이를 보기 위해, 구분자 없이 concat한 텍스트로 비교
    pred_concat = "".join(pred_norm)
    gold_concat = "".join(gold_norm)

    if pred_concat == gold_concat:
        return "split_merge_boundary"

    if strip_all_ws(pred_concat) == strip_all_ws(gold_concat):
        # 같은 텍스트인데 공백/줄바꿈 차이로 리스트가 다르게 보이는 케이스
        return "whitespace_only"

    # 합친 텍스트가 비슷한데 일부 문자만 다른 케이스(문장부호 등)
    ratio = SequenceMatcher(None, pred_concat, gold_concat).ratio()
    if ratio >= 0.98:
        return "minor_punct_or_typo"

    if len(pred_sents) != len(gold_sents):
        return "missing_or_extra_sentence"

    return "sentence_text_changed"


def iter_paragraph_sent_lists(df: pd.DataFrame) -> dict[tuple[str, int], list[str]]:
    # 파일 내 순서를 유지하기 위해 index 기반으로 정렬
    df = df.reset_index(drop=False).rename(columns={"index": "__row"})
    key_cols = ["book_name", "문단식별자"]
    if not all(c in df.columns for c in key_cols):
        raise ValueError(f"missing key columns: {key_cols}")
    if "번역문" not in df.columns:
        raise ValueError("missing column: 번역문")

    grouped: dict[tuple[str, int], list[str]] = {}
    for (book, pid), g in df.groupby(key_cols, sort=False):
        g = g.sort_values("__row")
        grouped[(str(book), int(pid))] = [str(x) for x in g["번역문"].tolist()]
    return grouped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("test_results/multitest_seed1_10/20260104_160112"),
        help="pa_output/gold csv들이 있는 run_dir",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="seed 목록(기본 1~10)",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/tgt_mismatch_seed1_10.md"),
        help="요약/예시 마크다운 출력 경로",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/tgt_mismatch_seed1_10.csv"),
        help="케이스 전체 CSV 출력 경로",
    )
    ap.add_argument(
        "--max-examples-per-type",
        type=int,
        default=6,
        help="유형별 예시 최대 개수",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    cases: list[MismatchCase] = []

    for seed in args.seeds:
        pred_path = run_dir / f"pa_output_n100_seed{seed}.csv"
        gold_path = run_dir / f"pa_gold_subset_n100_seed{seed}.csv"
        if not pred_path.exists() or not gold_path.exists():
            raise FileNotFoundError(f"missing: {pred_path} or {gold_path}")

        pred_df = pd.read_csv(pred_path)
        gold_df = pd.read_csv(gold_path)

        pred_map = iter_paragraph_sent_lists(pred_df)
        gold_map = iter_paragraph_sent_lists(gold_df)

        # key union으로 mismatch를 잡되, 분석/설명에는 gold 기준을 우선
        all_keys = sorted(set(pred_map.keys()) | set(gold_map.keys()))

        for (book, pid) in all_keys:
            pred_sents = pred_map.get((book, pid), [])
            gold_sents = gold_map.get((book, pid), [])

            if pred_sents == gold_sents:
                continue

            cat = classify(pred_sents, gold_sents)
            cases.append(
                MismatchCase(
                    seed=int(seed),
                    book_name=str(book),
                    paragraph_id=int(pid),
                    category=cat,
                    pred_sents=pred_sents,
                    gold_sents=gold_sents,
                )
            )

    # CSV dump
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "seed",
                "book_name",
                "paragraph_id",
                "category",
                "pred_count",
                "gold_count",
                "pred_join",
                "gold_join",
            ]
        )
        for c in cases:
            w.writerow(
                [
                    c.seed,
                    c.book_name,
                    c.paragraph_id,
                    c.category,
                    len(c.pred_sents),
                    len(c.gold_sents),
                    "\n".join(c.pred_sents),
                    "\n".join(c.gold_sents),
                ]
            )

    # Markdown summary
    counter = Counter([c.category for c in cases])
    by_type: dict[str, list[MismatchCase]] = defaultdict(list)
    for c in cases:
        by_type[c.category].append(c)

    # 예시는 재현성이 있도록 (seed, book, pid) 정렬
    for k in list(by_type.keys()):
        by_type[k] = sorted(by_type[k], key=lambda x: (x.seed, x.book_name, x.paragraph_id))

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with args.out_md.open("w", encoding="utf-8") as f:
        f.write("# 번역문(pred↔gold) 불일치 케이스 요약\n\n")
        f.write(f"- run_dir: `{run_dir.as_posix()}`\n")
        f.write(f"- seeds: `{args.seeds}`\n")
        f.write(f"- total mismatch paragraphs: **{len(cases)}**\n\n")

        f.write("## 유형별 카운트\n\n")
        for cat, n in counter.most_common():
            f.write(f"- {cat}: {n}\n")

        f.write("\n## 유형 정의(간단)\n\n")
        f.write("- normalization_only: 문장 리스트는 다르지만 정규화하면 동일\n")
        f.write("- split_merge_boundary: 문장 분할/병합 경계 차이(구분자 없이 합친 텍스트는 동일)\n")
        f.write("- whitespace_only: 공백/줄바꿈 차이로 동일 텍스트가 달라 보임\n")
        f.write("- minor_punct_or_typo: 거의 동일(문장부호/미세 오타 수준)\n")
        f.write("- missing_or_extra_sentence: 문장 수가 달라지고 내용도 달라짐\n")
        f.write("- sentence_text_changed: 문장 수는 같지만 텍스트 자체가 다름\n\n")

        f.write("## 예시\n\n")
        for cat, items in sorted(by_type.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            f.write(f"### {cat} (n={len(items)})\n\n")
            for c in items[: args.max_examples_per_type]:
                f.write(
                    f"- seed={c.seed} book={c.book_name} pid={c.paragraph_id} (pred={len(c.pred_sents)} gold={len(c.gold_sents)})\n"
                )
                f.write("\t- pred:\n")
                for s in c.pred_sents:
                    f.write(f"\t\t- {s}\n")
                f.write("\t- gold:\n")
                for s in c.gold_sents:
                    f.write(f"\t\t- {s}\n")
            f.write("\n")

    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_md}")
    print("categories:")
    for k, v in counter.most_common():
        print(" ", k, v)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
