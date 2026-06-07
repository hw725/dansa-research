#!/usr/bin/env python3
"""Recompute current cleaned and supplemented 3-model statistics.

The active quantitative baseline is the current cleaned dataset:

- section 1, section 2, and section 3 control supplements appended
- three-model consensus recomputed from active result CSVs

Outputs:
- results/cleaned_balanced_stats.json
- results/final_stats_v3.1_cleaned_balanced.json
- results/truth_tables_v3.1_cleaned_balanced.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

# Data source for judgment CSVs:
#   raw  = section*_judgments.csv (로컬 전용, 미공개 번역문 포함)
#   anon = section*_judgments_anon.csv (추적 대상, 번역문 해시)
#   auto = raw가 있으면 raw, 없으면 anon (새 클론은 anon으로 재현)
SOURCE = "auto"

MODELS = OrderedDict(
    [
        ("gpt5mini", "GPT-5-mini"),
        ("gemini", "Gemini 2.5 Flash"),
        ("claude_sonnet", "Claude Sonnet 4.6"),
    ]
)

KEY_COLS = ("book", "문단식별자", "문장식별자", "marker_type")

SECTIONS = OrderedDict(
    [
        (
            "section1",
            {
                "label": "游辭以斷",
                "csv": "section1_judgments.csv",
                "supplement": "supplement_section1_judgments.csv",
                "target": "로다",
                "control": "라(대조군)",
            },
        ),
        (
            "section2",
            {
                "label": "夬絶之斷 vs 微絶之斷",
                "csv": "section2_decision_judgments.csv",
                "supplement": "supplement_section2_judgments.csv",
                "target": "니라",
                "control": "라",
            },
        ),
        (
            "section3",
            {
                "label": "汎論以斷",
                "csv": "section3_judgments.csv",
                "supplement": "supplement_section3_judgments.csv",
                "target": "하나니라",
                "control": "라(대조군)",
            },
        ),
    ]
)


def parse_bool(value: str | bool | int | None) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "o"}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row.get(col, "") for col in KEY_COLS)


def dedupe(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    by_key: OrderedDict[tuple[str, str, str, str], dict[str, str]] = OrderedDict()
    for row in rows:
        by_key[row_key(row)] = row
    return list(by_key.values())


def resolve_csv(model_dir: Path, name: str) -> Path:
    """활성 SOURCE 모드에 따라 raw 또는 익명 판정 CSV 경로를 고른다."""
    raw = model_dir / name
    anon = model_dir / f"{Path(name).stem}_anon{Path(name).suffix}"
    if SOURCE == "raw":
        return raw
    if SOURCE == "anon":
        return anon
    return raw if raw.exists() else anon


def load_section_rows(model: str, cfg: dict[str, str | None]) -> list[dict[str, str]]:
    model_dir = RESULTS / model
    rows = read_csv_rows(resolve_csv(model_dir, str(cfg["csv"])))
    supplement = cfg.get("supplement")
    if supplement:
        rows.extend(read_csv_rows(resolve_csv(model_dir, str(supplement))))
    return dedupe(rows)


def chi_square_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    n = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if n == 0 or denom == 0:
        return 0.0, 1.0
    chi2 = n * ((a * d - b * c) ** 2) / denom
    p_value = math.erfc(math.sqrt(chi2 / 2))
    return chi2, p_value


def chi_square_table(rows: list[list[int]]) -> float:
    total = sum(sum(row) for row in rows)
    if total == 0:
        return 0.0
    row_totals = [sum(row) for row in rows]
    col_totals = [sum(rows[i][j] for i in range(len(rows))) for j in range(len(rows[0]))]
    chi2 = 0.0
    for i, row in enumerate(rows):
        for j, observed in enumerate(row):
            expected = row_totals[i] * col_totals[j] / total
            if expected:
                chi2 += ((observed - expected) ** 2) / expected
    return chi2


def pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def pct1(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 1) if denominator else 0.0


def per_model_stats(rows: list[dict[str, str]], target: str, control: str) -> dict[str, float | int | bool]:
    target_rows = [r for r in rows if r.get("marker_type") == target]
    control_rows = [r for r in rows if r.get("marker_type") == control]
    target_positive = sum(parse_bool(r.get("llm_judgment")) for r in target_rows)
    control_positive = sum(parse_bool(r.get("llm_judgment")) for r in control_rows)
    n_target = len(target_rows)
    n_control = len(control_rows)
    chi2, p_value = chi_square_2x2(
        target_positive,
        n_target - target_positive,
        control_positive,
        n_control - control_positive,
    )
    return {
        "target_n": n_target,
        "control_n": n_control,
        "target_rate": pct(target_positive, n_target),
        "control_rate": pct(control_positive, n_control),
        "diff": round(pct(target_positive, n_target) - pct(control_positive, n_control), 2),
        "target_positive": int(target_positive),
        "control_positive": int(control_positive),
        "chi2": round(chi2, 2),
        "p_approx": p_value,
    }


def consensus_stats(model_rows: dict[str, list[dict[str, str]]], target: str, control: str) -> dict:
    keyed: dict[str, dict[tuple[str, str, str, str], dict[str, str]]] = {}
    for model, rows in model_rows.items():
        keyed[model] = {row_key(row): row for row in rows}

    common_keys = set.intersection(*(set(v.keys()) for v in keyed.values()))
    counts = {
        target: {"O": 0, "S": 0, "X": 0, "n": 0},
        control: {"O": 0, "S": 0, "X": 0, "n": 0},
    }
    for key in common_keys:
        marker_type = key[3]
        if marker_type not in counts:
            continue
        votes = sum(parse_bool(keyed[model][key].get("llm_judgment")) for model in MODELS)
        counts[marker_type]["n"] += 1
        if votes == len(MODELS):
            counts[marker_type]["O"] += 1
        elif votes == 0:
            counts[marker_type]["X"] += 1
        else:
            counts[marker_type]["S"] += 1

    t = counts[target]
    c = counts[control]
    chi2 = chi_square_table([[t["O"], t["S"], t["X"]], [c["O"], c["S"], c["X"]]])
    total_n = t["n"] + c["n"]

    def block(group: dict[str, int]) -> dict[str, int | float]:
        n = group["n"]
        return {
            "O": group["O"],
            "O_pct": pct1(group["O"], n),
            "S": group["S"],
            "S_pct": pct1(group["S"], n),
            "X": group["X"],
            "X_pct": pct1(group["X"], n),
        }

    return {
        "target_n": t["n"],
        "control_n": c["n"],
        "target": block(t),
        "control": block(c),
        "chi2": round(chi2, 2),
        "V": round(math.sqrt(chi2 / total_n), 3) if total_n else 0,
        "N": total_n,
    }


def truth_table(section: str, cfg: dict, consensus: dict) -> dict:
    target = consensus["target"]
    control = consensus["control"]
    chi2, p_value = chi_square_2x2(
        target["O"],
        consensus["target_n"] - target["O"],
        control["O"],
        consensus["control_n"] - control["O"],
    )
    return {
        "section": section,
        "target_mt": cfg["target"],
        "control_mt": cfg["control"],
        "target": {"n": consensus["target_n"], **target},
        "control": {"n": consensus["control_n"], **control},
        "diff_pp": round(target["O_pct"] - control["O_pct"], 1),
        "chi2": round(chi2, 2),
        "p": p_value,
        "test": "chi2_no_correction",
    }


def build_payload() -> tuple[dict, dict]:
    sections_out: OrderedDict[str, dict] = OrderedDict()
    truth_tables: OrderedDict[str, dict] = OrderedDict()

    for section, cfg in SECTIONS.items():
        rows_by_model = {model: load_section_rows(model, cfg) for model in MODELS}
        consensus = consensus_stats(rows_by_model, str(cfg["target"]), str(cfg["control"]))
        per_model = OrderedDict()
        for model, display in MODELS.items():
            per_model[model] = {"display": display, **per_model_stats(rows_by_model[model], str(cfg["target"]), str(cfg["control"]))}
        sections_out[section] = {
            "label": cfg["label"],
            "target_marker": cfg["target"],
            "control_marker": cfg["control"],
            "consensus": consensus,
            "perModel": per_model,
        }
        truth_tables[section] = truth_table(section, cfg, consensus)

    return sections_out, truth_tables


def run_check() -> int:
    """현재 SOURCE로 통계를 재계산해 기준 JSON과 대조한다 (파일 미기록)."""
    sections_out, _ = build_payload()
    ref_path = RESULTS / "final_stats_v3.1_cleaned_balanced.json"
    if not ref_path.exists():
        print("기준 JSON 없음: final_stats_v3.1_cleaned_balanced.json")
        return 1
    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)["sections"]
    ok = True
    for section, data in sections_out.items():
        computed = data["consensus"]
        reference = ref.get(section, {}).get("consensus", {})
        for field in ("target_n", "control_n", "chi2", "V", "N"):
            if computed.get(field) != reference.get(field):
                ok = False
                print(f"  [DIFF] {section}.{field}: 계산={computed.get(field)} 기준={reference.get(field)}")
    print(f"CHECK {'PASS' if ok else 'FAIL'} (source={SOURCE})")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    global SOURCE
    parser = argparse.ArgumentParser(description="3-model 통계 재계산")
    parser.add_argument("--source", choices=["auto", "raw", "anon"], default="auto",
                        help="판정 CSV 소스 (기본 auto: raw 없으면 anon)")
    parser.add_argument("--check", action="store_true",
                        help="기준 JSON과 대조만 하고 파일을 쓰지 않는다")
    args = parser.parse_args(argv)
    SOURCE = args.source

    if args.check:
        return run_check()

    sections_out, truth_tables = build_payload()
    today = dt.date.today().isoformat()
    metadata = {
        "generated_at": today,
        "manuscript_version": "v3.1_cleaned_balanced",
        "basis": "Cleaned May 20 data plus section 1/2/3 control supplements",
        "sections": sections_out,
    }

    targets = {
        RESULTS / "cleaned_balanced_stats.json": sections_out,
        RESULTS / "truth_tables_v3.1_cleaned_balanced.json": truth_tables,
        RESULTS / "final_stats_v3.1_cleaned_balanced.json": metadata,
    }
    for path, payload in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
