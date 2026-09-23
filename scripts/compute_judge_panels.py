#!/usr/bin/env python3
"""판정자 구성을 바꿔 논문 산출물을 그대로 다시 낸다(호출 0건).

왜:
    논문 수치(final_stats·truth_tables·강건성 보고서)는 3모델 판정에서 나왔다. 판정 모델
    Jev·Solar를 더하면 같은 산출물이 어떻게 바뀌는지 봐야 한다 — 새 지표를 만들지 않고
    **기존 스크립트의 함수를 그대로** 부른다. 바꾸는 것은 `cfs.MODELS`(판정자 목록)뿐이다.

판정자:
    3모델은 기존 판정 CSV, Jev·Solar는 `run_jev_judgments.py --order single`의 결과
    (results/jev/single/*.jsonl)다. 확률 ≥ --threshold(기본 0.5)를 O로 본다 — 3모델의 O/X와
    같은 이진 판정으로 만든다.

구성(panel):
    3models          기준. 정본(final_stats_v3.1)과 같아야 한다 — 이 생성기의 검산이다.
    jev_solar        두 판정 모델의 합의. O = 둘 다 O, X = 둘 다 X, S = 갈라짐.
    3models_jev / 3models_solar  3모델에 판정 모델 하나를 더한 네 판정자(만장일치 4표, 과반 3표).
    3models_jev_solar 다섯 판정자. 만장일치는 5표.
    판정 결과가 없는 판정자가 끼면 그 구성은 건너뛴다. 결과가 일부만 있으면 공통 문항만
    쓰고 coverage에 적는다(부분 결과를 전체처럼 읽지 않도록).

출력: results/jev/panels/<panel>/{final_stats,truth_tables,robustness_stats}.json ·
      ROBUSTNESS_REPORT.md, 그리고 results/jev/panels/SUMMARY.md(구성 대조표).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_final_stats as cfs  # noqa: E402
import compute_robustness_stats as rb  # noqa: E402
import run_jev_judgments as rj  # noqa: E402

OUT = rj.OUT_DIR / "panels"
BASE_MODELS = OrderedDict(cfs.MODELS)
BASE_SECTIONS = OrderedDict(cfs.SECTIONS)
JUDGES = OrderedDict([
    ("jev", ("jev-latest", "Jev (jev-latest)")),
    ("solar", ("solar-mini4-jev", "Solar Mini4 (solar-mini4-jev)")),
])
PANELS = OrderedDict([
    ("3models", list(BASE_MODELS)),
    ("jev_solar", ["jev", "solar"]),
    ("3models_jev", [*BASE_MODELS, "jev"]),
    ("3models_solar", [*BASE_MODELS, "solar"]),
    ("3models_jev_solar", [*BASE_MODELS, "jev", "solar"]),
])


def judge_rows(judge: str, section: str, threshold: float, order_name: str) -> list[dict]:
    """판정 모델 결과 → 3모델 판정 CSV와 같은 모양의 행(키 4열 + llm_judgment)."""
    model = JUDGES[judge][0]
    path = rj.out_path(section, "pos", model, order_name)
    rows: OrderedDict = OrderedDict()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("noul") is None:
                continue
            k = rec["key"]
            rows[tuple(k)] = {"book": k[0], "문단식별자": k[1], "문장식별자": k[2],
                              "marker_type": k[3],
                              "llm_judgment": str(float(rec["noul"]) >= threshold)}
    return list(rows.values())


def install_loader(threshold: float, order_name: str) -> None:
    """cfs.load_section_rows를 판정 모델도 읽도록 감싼다. 두 스크립트 모두 모듈 속성으로 부르므로
    여기서 바꾸면 build_payload·collect_items 양쪽에 듣는다."""
    orig = cfs.load_section_rows
    by_csv = {cfg["csv"]: sec for sec, cfg in BASE_SECTIONS.items()}

    def load(model: str, cfg: dict) -> list[dict]:
        if model in JUDGES:
            return judge_rows(model, by_csv[cfg["csv"]], threshold, order_name)
        return orig(model, cfg)

    cfs.load_section_rows = load


def coverage(judges: list[str]) -> dict[str, dict]:
    """섹션마다 3모델 문항 중 이 구성 전원이 판정한 비율."""
    out = {}
    for sec, cfg in BASE_SECTIONS.items():
        keysets = [{cfs.row_key(r) for r in cfs.load_section_rows(m, cfg)} for m in judges]
        base = {cfs.row_key(r) for r in cfs.load_section_rows("gpt5mini", cfg)}
        common = set.intersection(*keysets) & base if keysets else set()
        out[sec] = {"n_base": len(base), "n_common": len(common),
                    "complete": len(common) == len(base)}
    return out


def run_panel(name: str, judges: list[str], boot: int, seed: int, allow_partial: bool) -> dict | None:
    cov = coverage(judges)
    sections = OrderedDict((s, c) for s, c in BASE_SECTIONS.items()
                           if cov[s]["n_common"] and (cov[s]["complete"] or allow_partial))
    if not sections:
        print(f"[{name}] 판정 결과가 없어 건너뜁니다.")
        return None
    cfs.MODELS = OrderedDict((j, BASE_MODELS.get(j) or JUDGES[j][1]) for j in judges)
    cfs.SECTIONS = sections
    try:
        final_sections, truth = cfs.build_payload()
        rng = random.Random(seed)
        robust = OrderedDict((s, rb.analyze_section(cfg, boot, rng)) for s, cfg in sections.items())
    finally:
        cfs.MODELS, cfs.SECTIONS = OrderedDict(BASE_MODELS), OrderedDict(BASE_SECTIONS)

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {"generated_at": dt.date.today().isoformat(), "panel": name, "judges": judges,
            "coverage": {s: cov[s] for s in sections}}
    (d / "final_stats.json").write_text(json.dumps({**meta, "sections": final_sections},
                                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (d / "truth_tables.json").write_text(json.dumps({**meta, "sections": truth},
                                         ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    rpayload = {"generated_at": meta["generated_at"], "source": "panel", "boot": boot, "seed": seed,
                "basis": f"판정자 {', '.join(judges)}", "sections": robust}
    (d / "robustness_stats.json").write_text(json.dumps({**meta, **rpayload}, ensure_ascii=False, indent=2)
                                             + "\n", encoding="utf-8", newline="\n")
    cfs.MODELS = OrderedDict((j, BASE_MODELS.get(j) or JUDGES[j][1]) for j in judges)
    try:
        report = rb.render_report(rpayload)
    finally:
        cfs.MODELS = OrderedDict(BASE_MODELS)
    head = (f"> 판정자 구성 **{name}**: {', '.join(judges)}. `scripts/compute_judge_panels.py`가 "
            f"`compute_robustness_stats.py`의 함수를 그대로 불러 낸 것이다. 아래 본문의 «새 LLM 호출 없음» "
            f"문구는 원 보고서의 것이다 — Jev·Solar 판정은 run_jev_judgments.py로 따로 받았다.\n\n")
    (d / "ROBUSTNESS_REPORT.md").write_text(head + report, encoding="utf-8", newline="\n")
    return {"final": final_sections, "robust": robust, "coverage": meta["coverage"], "judges": judges}


def check_baseline(res: dict) -> list[str]:
    """3models 구성이 정본과 같은가 — 생성기 자체의 검산."""
    ref = json.loads((cfs.RESULTS / "final_stats_v3.1_cleaned_balanced.json").read_text(encoding="utf-8"))["sections"]
    diffs = []
    for s, data in res["final"].items():
        for f in ("target_n", "control_n", "chi2", "V", "N"):
            if data["consensus"].get(f) != ref[s]["consensus"].get(f):
                diffs.append(f"{s}.{f}: {data['consensus'].get(f)} ≠ {ref[s]['consensus'].get(f)}")
    return diffs


def summary(results: dict) -> str:
    L = ["# 판정자 구성별 논문 산출물 대조", "",
         f"생성 {dt.date.today().isoformat()} · `scripts/compute_judge_panels.py` · 판정 모델은 한 요청에 한 문장(single), "
         "확률 ≥ 0.5를 O로 본다.", ""]
    for s, cfg in BASE_SECTIONS.items():
        present = [(p, r) for p, r in results.items() if r and s in r["final"]]
        if not present:
            continue
        L += [f"## {s} {cfg['label']} ({cfg['target']} 대 {cfg['control']})", ""]
        L += ["### 합의 O/S/X (O = 전원 O, X = 전원 X, S = 갈라짐)", "",
              "| 구성 | n | target O / S / X % | control O / S / X % | χ² | V | 범위 |",
              "|---|---:|---|---|---:|---:|---|"]
        for p, r in present:
            c = r["final"][s]["consensus"]
            t, k = c["target"], c["control"]
            cov = r["coverage"][s]
            scope = "전체" if cov["complete"] else f"부분 {cov['n_common']}/{cov['n_base']}"
            L.append(f"| {p} | {c['N']} | {t['O_pct']} / {t['S_pct']} / {t['X_pct']} | "
                     f"{k['O_pct']} / {k['S_pct']} / {k['X_pct']} | {c['chi2']} | {c['V']} | {scope} |")
        wide = next((r for p, r in reversed(present)), None)
        L += ["", "### 판정자별 O 비율", "", "| 판정자 | target O% | control O% | 차 %p | χ² |", "|---|---:|---:|---:|---:|"]
        seen = set()
        for p, r in present:
            for m, pm in r["final"][s]["perModel"].items():
                if m in seen:
                    continue
                seen.add(m)
                L.append(f"| {pm['display']} | {pm['target_rate']} | {pm['control_rate']} | {pm['diff']} | {pm['chi2']} |")
        L += ["", "### 판정자 쌍 일치 (가장 넓은 구성 기준)", "", "| 쌍 | 일치 % | Cohen κ |", "|---|---:|---:|"]
        ag = wide["robust"][s]["agreement"]
        for pair, blk in ag["pairwise"].items():
            L.append(f"| {pair.replace('__', ' × ')} | {blk['percent_agreement']} | {blk['cohen_kappa']} |")
        L += ["", "### 강건성 (구성별)", "",
              "| 구성 | Fleiss κ | 만장일치 % | O율차 %p [95% CI] | OR [95% CI] | MH OR(책) |",
              "|---|---:|---:|---|---|---|"]
        for p, r in present:
            rs = r["robust"][s]
            ov = rs["interval_estimates"]["o_vs_rest"]
            orr = ov["odds_ratio"]
            mh = rs["stratified"]["mh_book"]
            L.append(f"| {p} | {rs['agreement']['fleiss_kappa']} | {rs['agreement']['unanimous_pct']} | "
                     f"{ov['diff_pp']} {ov.get('diff_ci95_pp')} | {orr['or']} {orr['ci95']} | "
                     f"{mh['or'] if mh else '—'} |")
        L += ["", "### 합의 정의 민감도 (구성별 O율차 %p)", "", "| 구성 | 만장일치 | 과반 | 1표 이상 |", "|---|---:|---:|---:|"]
        for p, r in present:
            sen = r["robust"][s]["sensitivity"]
            L.append(f"| {p} | {sen['unanimous']['diff_pp']} | {sen['majority']['diff_pp']} | {sen['any']['diff_pp']} |")
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--order", default="single", help="판정 모델 결과의 묶는 방식(run_jev_judgments --order)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260611, help="강건성 보고서와 같은 부트스트랩 시드")
    ap.add_argument("--allow-partial", action="store_true", help="판정이 일부만 있는 섹션도 공통 문항으로 낸다")
    ap.add_argument("--source", choices=["auto", "raw", "anon"], default="auto",
                    help="3모델 판정 CSV 소스(compute_final_stats와 같다). 공개 클론은 anon")
    a = ap.parse_args(argv)
    cfs.SOURCE = a.source
    install_loader(a.threshold, a.order)
    results = OrderedDict()
    for name, judges in PANELS.items():
        print(f"[{name}] …")
        results[name] = run_panel(name, judges, a.boot, a.seed, a.allow_partial)
    if results.get("3models"):
        diffs = check_baseline(results["3models"])
        print("기준(3models) 정본 대조:", "PASS" if not diffs else "FAIL " + "; ".join(diffs))
        if diffs:
            return 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.md").write_text(summary(results), encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
