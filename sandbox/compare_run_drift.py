"""compare_run_drift.py — 재실행 결과와 동결 기준 결과의 드리프트 리포트.

목적: 정확한 숫자 일치가 아니라 LLM 판정 효과가 유지되는지 확인하는 것이다.
세 가지를 비교한다.

1. 섹션별 Consensus O 비율(target/control)과 효과 방향 보존 여부
2. χ²·Cramér's V 의 변화량
3. 문장 단위 재현: 재실행 판정이 동결 익명 판정과 얼마나 일치하는가
   (per-model 일치율 + Cohen's κ)

동결 기준 final_stats 는 base+supplement 를 포함하지만 재실행은 LLM manifest(base)
표본만 다시 수행한다. 따라서 절대 건수(N)가 아니라 비율·효과 방향·문장 일치로 비교한다.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "scripts"))

import compute_final_stats as cfs  # noqa: E402  (consensus 계산 재사용)

A_RUN_DIR = Path(os.environ.get("A_RUN_DIR", "/tmp/a_run"))
FINAL = cfs.RESULTS / "final_stats_v3.1_cleaned_balanced.json"


def _load(model_dir: Path, name: str) -> list[dict]:
    return cfs.read_csv_rows(model_dir / name)


def _fresh_consensus(section_key: str, cfg: dict) -> dict | None:
    rows_by_model = {m: _load(A_RUN_DIR / m, cfg["csv"]) for m in cfs.MODELS}
    if any(not rows for rows in rows_by_model.values()):
        return None
    return cfs.consensus_stats(rows_by_model, str(cfg["target"]), str(cfg["control"]))


def _cohen_kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(1 for a, _ in pairs if a) / n
    pb1 = sum(1 for _, b in pairs if b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 0.0 if pe >= 1 else round((po - pe) / (1 - pe), 3)


def _sentence_agreement(section_key: str, cfg: dict) -> dict:
    """동결 익명 판정 vs 재실행 판정, 동일 키 기준 per-model 일치."""
    out = {}
    for model in cfs.MODELS:
        frozen = cfs.read_csv_rows(
            cfs.RESULTS / model / f"{Path(cfg['csv']).stem}_anon.csv")
        fresh = _load(A_RUN_DIR / model, cfg["csv"])
        fz = {cfs.row_key(r): cfs.parse_bool(r.get("llm_judgment")) for r in frozen}
        fr = {cfs.row_key(r): cfs.parse_bool(r.get("llm_judgment")) for r in fresh}
        common = set(fz) & set(fr)
        pairs = [(fz[k], fr[k]) for k in common]
        agree = round(sum(1 for a, b in pairs if a == b) / len(pairs) * 100, 1) if pairs else 0.0
        out[model] = {"n": len(pairs), "agree_pct": agree, "kappa": _cohen_kappa(pairs)}
    return out


def main() -> int:
    if not FINAL.exists():
        print(f"[drift] 동결 기준 없음: {FINAL}")
        return 1
    frozen = json.loads(FINAL.read_text(encoding="utf-8"))["sections"]

    lines: list[str] = []
    report: dict = {"sections": {}, "note": "rerun vs frozen reference — 비율·효과방향·문장일치"}
    directions_ok = 0
    sections_done = 0

    lines.append("# 재실행 드리프트 리포트 (재현본 vs 동결본)\n")
    lines.append("| 섹션 | 동결 O%(T/C) | 재실행 O%(T/C) | ΔV | 효과방향 |")
    lines.append("|---|---|---|---|---|")

    for sk, cfg in cfs.SECTIONS.items():
        fresh = _fresh_consensus(sk, cfg)
        fz = frozen.get(sk, {}).get("consensus", {})
        if fresh is None or not fz:
            lines.append(f"| {sk} | — | (재실행 없음) | — | — |")
            continue
        sections_done += 1
        fz_t, fz_c = fz["target"]["O_pct"], fz["control"]["O_pct"]
        fr_t, fr_c = fresh["target"]["O_pct"], fresh["control"]["O_pct"]
        dir_frozen = fz_t - fz_c
        dir_fresh = fr_t - fr_c
        same_dir = (dir_frozen > 0) == (dir_fresh > 0)
        directions_ok += int(same_dir)
        dv = round(fresh["V"] - fz["V"], 3)
        agree = _sentence_agreement(sk, cfg)
        lines.append(
            f"| {cfg['label']} | {fz_t}/{fz_c} | {fr_t}/{fr_c} | {dv:+} | "
            f"{'보존 ✓' if same_dir else '역전 ✗'} |")
        report["sections"][sk] = {
            "frozen": {"target_O_pct": fz_t, "control_O_pct": fz_c, "V": fz["V"], "chi2": fz["chi2"]},
            "a_run": {"target_O_pct": fr_t, "control_O_pct": fr_c, "V": fresh["V"], "chi2": fresh["chi2"]},
            "effect_direction_preserved": same_dir,
            "sentence_agreement": agree,
        }

    lines.append("\n## 문장 단위 재현 (동결 익명 판정 대비)\n")
    lines.append("| 섹션 | 모델 | 공통 n | 일치율 | κ |")
    lines.append("|---|---|---|---|---|")
    for sk, data in report["sections"].items():
        for model, a in data["sentence_agreement"].items():
            lines.append(f"| {sk} | {cfs.MODELS[model]} | {a['n']:,} | {a['agree_pct']}% | {a['kappa']} |")

    verdict = (f"\n## 판정\n효과 방향 재현: {directions_ok}/{sections_done} 섹션. "
               "절대 수치는 비결정성·모델 갱신으로 달라질 수 있으나, 방향·문장 일치가 "
               "유지되면 동결 기준 결론을 보조적으로 확인한 것이다.")
    lines.append(verdict)
    report["effect_direction_preserved"] = f"{directions_ok}/{sections_done}"

    text = "\n".join(lines)
    print("\n" + text + "\n")
    (A_RUN_DIR / "drift_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
