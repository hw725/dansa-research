"""섹션 2 보충 판정 반영 + 3모델 합의 → parallel_data_v2_cleaned.tsv 재생성.

데이터 흐름:
  results/{model}/section2_decision_judgments.csv        (원본)
  results/{model}/supplement_section2_judgments.csv      (보충)
  → 현재 sentence 코퍼스 범위 검증
  → 3모델 교차 → consensus(O/X/split)
  → cell 배정 (I~IV)
  → parallel 컬럼 생성
  → analysis/parallel_data_v2_cleaned.tsv
"""
import os, csv, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # dansa-research
RESULTS = ROOT / "results"
OUT_TSV = ROOT / "analysis" / "parallel_data_v2_cleaned.tsv"

MODELS = {
    "gpt5mini": RESULTS / "gpt5mini",
    "gemini": RESULTS / "gemini",
    "claude_sonnet": RESULTS / "claude_sonnet",
}

ZZTJGM_PREFIX = "자치통감강목"
MAX_CURRENT_ZZTJGM_VOLUME = 7


def is_out_of_scope_book(book: str) -> bool:
    if not book.startswith(ZZTJGM_PREFIX):
        return False
    suffix = book.removeprefix(ZZTJGM_PREFIX)
    return suffix.isdigit() and int(suffix) > MAX_CURRENT_ZZTJGM_VOLUME


def load_section2(model_dir: Path) -> dict:
    """원본 + 보충 섹션 2 판정을 읽어 {(book, 문장식별자): row} 반환."""
    rows = {}
    for fname in ["section2_decision_judgments.csv", "supplement_section2_judgments.csv"]:
        fpath = model_dir / fname
        if not fpath.exists():
            print(f"  [skip] {fpath}")
            continue
        with open(fpath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                book = r["book"]
                if is_out_of_scope_book(book):
                    raise ValueError(f"out-of-scope book in current input: {book}")
                key = (book, r["문장식별자"])
                # 보충 판정이 기존을 덮어쓸 수 있음 (같은 키면 최신 우선)
                rows[key] = r
    return rows


def main():
    # 1. 각 모델별 섹션 2 판정 로드
    model_data = {}
    for name, mdir in MODELS.items():
        print(f"Loading {name}...")
        model_data[name] = load_section2(mdir)
        print(f"  {len(model_data[name])} rows")

    # 2. 3모델 공통 키 찾기
    keys_sets = [set(d.keys()) for d in model_data.values()]
    common_keys = keys_sets[0]
    for ks in keys_sets[1:]:
        common_keys = common_keys & ks
    print(f"\n3-model common keys: {len(common_keys)}")

    # 니라/라 분리 확인
    nira_keys = set()
    ra_keys = set()
    # 아무 모델이나 참조 (marker_type은 모델 무관)
    ref = model_data["gpt5mini"]
    for k in common_keys:
        mt = ref[k].get("marker_type", ref[k].get("marker_normalized", ""))
        if mt == "니라":
            nira_keys.add(k)
        else:
            ra_keys.add(k)
    print(f"  니라: {len(nira_keys)}, 라: {len(ra_keys)}")

    # 3. 3모델 합의 계산
    consensus_rows = []
    model_names = list(MODELS.keys())
    stats = defaultdict(int)

    for k in sorted(common_keys):
        judgments = []
        for mname in model_names:
            j = model_data[mname][k]["llm_judgment"]
            judgments.append(str(j).strip().lower() == "true")

        if all(judgments):
            decision = "O"
        elif not any(judgments):
            decision = "X"
        else:
            decision = "split"
            stats["split"] += 1
            continue  # split은 TSV에서 제외

        # 기준 모델에서 메타데이터 가져오기
        r = ref[k]
        mt = r.get("marker_type", r.get("marker_normalized", ""))

        if mt == "니라":
            if decision == "O":
                cell = "I_니라_O"
            else:
                cell = "II_니라_X"
        else:
            if decision == "O":
                cell = "III_라_O"
            else:
                cell = "IV_라_X"

        stats[cell] += 1

        parallel = f"{r['원문']} ||| {r['번역문']}"

        consensus_rows.append({
            "cell": cell,
            "book": r["book"],
            "문단식별자": r["문단식별자"],
            "문장식별자": r["문장식별자"],
            "marker_raw": r["marker_raw"],
            "marker_type": mt,
            "marker_normalized": r.get("marker_normalized", mt),
            "dansa_category": r["dansa_category"],
            "원문": r["원문"],
            "번역문": r["번역문"],
            "parallel": parallel,
        })

    print(f"\n=== Consensus Results ===")
    for cell in ["I_니라_O", "II_니라_X", "III_라_O", "IV_라_X"]:
        print(f"  {cell}: {stats[cell]}")
    print(f"  split (excluded): {stats['split']}")
    print(f"  Total consensus: {sum(stats[c] for c in ['I_니라_O','II_니라_X','III_라_O','IV_라_X'])}")

    nira_total = len(nira_keys)
    ra_total = len(ra_keys)
    nira_consensus = stats["I_니라_O"] + stats["II_니라_X"]
    ra_consensus = stats["III_라_O"] + stats["IV_라_X"]
    print(f"\n  니라 total: {nira_total}, consensus: {nira_consensus}, split: {nira_total - nira_consensus} ({(nira_total-nira_consensus)/nira_total*100:.1f}%)")
    print(f"  라 total: {ra_total}, consensus: {ra_consensus}, split: {ra_total - ra_consensus} ({(ra_total-ra_consensus)/ra_total*100:.1f}%)")

    # 4. TSV 출력
    fieldnames = ["cell", "book", "문단식별자", "문장식별자", "marker_raw",
                   "marker_type", "marker_normalized", "dansa_category",
                   "원문", "번역문", "parallel"]

    with open(OUT_TSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in consensus_rows:
            writer.writerow(row)

    print(f"\nSaved: {OUT_TSV} ({len(consensus_rows)} rows)")


if __name__ == "__main__":
    main()
