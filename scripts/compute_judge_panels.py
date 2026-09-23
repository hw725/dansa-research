#!/usr/bin/env python3
"""판정자 구성을 바꿔 논문 산출물을 그대로 다시 낸다(호출 0건).

왜:
    논문 수치(final_stats·truth_tables·강건성 보고서)는 3모델 판정에서 나왔다. 판정 모델
    Jev·Solar를 더하면 같은 산출물이 어떻게 바뀌는지 봐야 한다 — 새 지표를 만들지 않고
    **기존 스크립트의 함수를 그대로** 부른다.

무엇을 바꾸는가(이것뿐이다):
    - 판정자 목록 `cfs.MODELS`와 낼 섹션 `cfs.SECTIONS`
    - 행을 읽는 함수 `cfs.load_section_rows` — 판정 모델의 JSONL도 읽도록 감싼다. 두 통계
      스크립트가 이 함수를 모듈 속성으로 부르기 때문에 여기서 바꾸면 양쪽에 듣는다.
    - 판정 모델의 확률을 O/X로 바꾼다(--threshold, 기본 0.5 이상 = O). 3모델의 O/X와 같은
      이진 판정이 된다.
    바꾼 전역 값은 구성마다, 그리고 끝날 때 원래대로 되돌린다.

표본:
    기준은 **3모델 공통 문항 중 target·control 표지인 것**(run_jev_judgments가 묻는 문항과
    같다). 판정 모델 행도 이 문항으로 제한한다. 구성 전원이 기준 문항을 다 판정한 섹션만
    낸다. --allow-partial이면 일부만 판정한 섹션도 내는데, 그때는 **모든 판정자의 행을 공통
    문항으로 제한**하고(판정자별 비율까지 같은 표본), 범위를 산출물에 적는다.

구성(panel):
    3models          기준. 정본과 **통째로** 대조한다(final_stats의 섹션 전체, 세 섹션이 다
                     있으면 강건성 JSON의 섹션 전체).
    jev_solar        두 판정 모델의 합의. O = 둘 다 O, X = 둘 다 X, S = 갈라짐.
    3models_jev / 3models_solar  네 판정자(만장일치 4표, 과반 3표 이상).
    3models_jev_solar 다섯 판정자(만장일치 5표, 과반 3표 이상).

출력: results/jev/panels/<panel>/{final_stats,truth_tables,robustness_stats}.json ·
      ROBUSTNESS_REPORT.md, 그리고 results/jev/panels/SUMMARY.md(구성 대조표).
      JSON에는 threshold·order·source·표본 범위를 함께 적는다.
"""
from __future__ import annotations

import argparse
import contextlib
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


class Loader:
    """cfs.load_section_rows 대체. 판정 모델 행을 만들고, 필요하면 모든 판정자의 행을 공통
    문항으로 거른다. install()/restore()로 원래 함수를 되돌린다."""

    def __init__(self, threshold: float, order_name: str) -> None:
        if order_name not in rj.ORDERS:
            raise ValueError(f"--order는 {rj.ORDERS} 중 하나여야 합니다: {order_name!r}")
        self.threshold = threshold
        self.order_name = order_name
        self.orig = cfs.load_section_rows
        self.by_csv = {cfg["csv"]: sec for sec, cfg in BASE_SECTIONS.items()}
        self.base: dict[str, set] = {}      # 섹션 → 기준 문항 키
        self.only: dict[str, set] = {}      # 섹션 → 이 구성에서 쓸 문항(부분 섹션만)
        self.skipped_lines: dict[str, int] = {}

    def install(self) -> None:
        # 기준 문항은 원래 함수·3모델로 먼저 센다(rj._load가 cfs.MODELS 전원의 공통 키를 쓴다).
        for sec in BASE_SECTIONS:
            self.base[sec] = {tuple(it["key"]) for it in rj._load(sec, need_text=False)}
        cfs.load_section_rows = self.load

    def restore(self) -> None:
        cfs.load_section_rows = self.orig

    def judge_rows(self, judge: str, section: str) -> list[dict]:
        """판정 모델 JSONL → 3모델 판정 CSV와 같은 모양의 행(키 4열 + llm_judgment)."""
        path = rj.out_path(section, "pos", JUDGES[judge][0], self.order_name)
        recs, bad = rj.read_jsonl(path)
        if bad:
            self.skipped_lines[f"{judge}:{section}"] = bad
        rows: OrderedDict = OrderedDict()
        for rec in recs:
            k = tuple(rec["key"])
            if rec.get("noul") is None or k not in self.base[section]:
                continue
            rows[k] = {"book": k[0], "문단식별자": k[1], "문장식별자": k[2], "marker_type": k[3],
                       "llm_judgment": str(float(rec["noul"]) >= self.threshold)}
        return list(rows.values())

    def load(self, model: str, cfg: dict) -> list[dict]:
        section = self.by_csv[cfg["csv"]]
        rows = self.judge_rows(model, section) if model in JUDGES else self.orig(model, cfg)
        keep = self.only.get(section)
        return rows if keep is None else [r for r in rows if cfs.row_key(r) in keep]


@contextlib.contextmanager
def panel_state(models: OrderedDict, sections: OrderedDict, loader: Loader, only: dict):
    """판정자·섹션·표본 제한을 바꿨다가 **들어오기 전 값으로** 되돌린다(예외여도)."""
    saved = (cfs.MODELS, cfs.SECTIONS, loader.only)
    cfs.MODELS, cfs.SECTIONS, loader.only = models, sections, only
    try:
        yield
    finally:
        cfs.MODELS, cfs.SECTIONS, loader.only = saved


def coverage(judges: list[str], loader: Loader) -> dict[str, dict]:
    """섹션마다 기준 문항 중 이 구성 전원이 판정한 문항."""
    out = {}
    for sec, cfg in BASE_SECTIONS.items():
        common = set(loader.base[sec])
        for m in judges:
            common &= {cfs.row_key(r) for r in loader.load(m, cfg)}
        out[sec] = {"n_base": len(loader.base[sec]), "n_common": len(common),
                    "complete": common == loader.base[sec], "keys": common}
    return out


def run_panel(name: str, judges: list[str], loader: Loader, meta_run: dict,
              boot: int, seed: int, allow_partial: bool) -> dict | None:
    cov = coverage(judges, loader)
    sections = OrderedDict((s, c) for s, c in BASE_SECTIONS.items()
                           if cov[s]["n_common"] and (cov[s]["complete"] or allow_partial))
    if not sections:
        print(f"[{name}] 판정 결과가 없어 건너뜁니다.")
        return None
    only = {s: cov[s]["keys"] for s in sections if not cov[s]["complete"]}
    models = OrderedDict((j, BASE_MODELS.get(j) or JUDGES[j][1]) for j in judges)
    with panel_state(models, sections, loader, only):
        final_sections, truth = cfs.build_payload()
        rng = random.Random(seed)  # 정본과 같다 — 섹션 순서대로 한 RNG를 이어 쓴다
        robust = OrderedDict((s, rb.analyze_section(cfg, boot, rng)) for s, cfg in sections.items())

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    cov_out = {s: {k: v for k, v in cov[s].items() if k != "keys"} for s in sections}
    meta = {"generated_at": dt.date.today().isoformat(), "panel": name, "judges": judges,
            **meta_run, "coverage": cov_out}
    for fname, payload in (("final_stats.json", {"sections": final_sections}),
                           ("truth_tables.json", {"sections": truth})):
        (d / fname).write_text(json.dumps({**meta, **payload}, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
    rpayload = {"generated_at": meta["generated_at"], "source": meta_run["source"], "boot": boot,
                "seed": seed, "basis": f"판정자 {', '.join(judges)}", "sections": robust}
    (d / "robustness_stats.json").write_text(json.dumps({**meta, **rpayload}, ensure_ascii=False, indent=2)
                                             + "\n", encoding="utf-8", newline="\n")
    with panel_state(models, sections, loader, only):
        report = rb.render_report(rpayload)
    head = (f"> 판정자 구성 **{name}**: {', '.join(judges)} · 판정 모델 O = 확률 ≥ {meta_run['threshold']} · "
            f"묶음 {meta_run['order']} · 3모델 소스 {meta_run['source']}. `scripts/compute_judge_panels.py`가 "
            f"`compute_robustness_stats.py`의 함수를 그대로 불러 낸 것이다. 아래 본문의 «새 LLM 호출 없음» "
            f"문구는 원 보고서의 것이다 — 판정 모델의 판정은 run_jev_judgments.py로 따로 받았다.\n\n")
    (d / "ROBUSTNESS_REPORT.md").write_text(head + report, encoding="utf-8", newline="\n")
    return {"final": final_sections, "robust": robust, "coverage": cov_out, "judges": judges}


def check_baseline(res: dict) -> list[str]:
    """3models 구성이 정본과 같은가 — final_stats는 섹션 전체(합의·판정자별)를, 강건성은
    세 섹션이 다 있을 때 섹션 전체를 비교한다."""
    diffs = []
    ref = json.loads((cfs.RESULTS / "final_stats_v3.1_cleaned_balanced.json").read_text(encoding="utf-8"))
    for s in BASE_SECTIONS:
        if s not in res["final"]:
            diffs.append(f"{s}: 3models 구성에 섹션이 없다")
        elif res["final"][s] != ref["sections"][s]:
            diffs.append(f"{s}: final_stats 섹션이 정본과 다르다")
    if list(res["robust"]) == list(BASE_SECTIONS):
        rref = json.loads((cfs.RESULTS / "robustness_stats.json").read_text(encoding="utf-8"))["sections"]
        for s in BASE_SECTIONS:
            if json.loads(json.dumps(res["robust"][s])) != rref[s]:
                diffs.append(f"{s}: robustness 섹션이 정본과 다르다")
    return diffs


def summary(results: dict, meta_run: dict) -> str:
    L = ["# 판정자 구성별 논문 산출물 대조", "",
         f"생성 {dt.date.today().isoformat()} · `scripts/compute_judge_panels.py` · 판정 모델 묶음 "
         f"{meta_run['order']}, 확률 ≥ {meta_run['threshold']}를 O로 본다 · 3모델 소스 {meta_run['source']}.", ""]
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
        L += ["", "### 판정자별 O 비율", "",
              "판정자마다 처음 나오는 구성의 값을 쓴다. 표본이 전체인 구성에서는 판정자별 값이 구성과 무관하다.", "",
              "| 판정자 | 출처 구성 | n (target / control) | target O% | control O% | 차 %p | χ² |",
              "|---|---|---|---:|---:|---:|---:|"]
        seen = set()
        for p, r in present:
            for m, pm in r["final"][s]["perModel"].items():
                if m in seen:
                    continue
                seen.add(m)
                L.append(f"| {pm['display']} | {p} | {pm['target_n']} / {pm['control_n']} | "
                         f"{pm['target_rate']} | {pm['control_rate']} | {pm['diff']} | {pm['chi2']} |")
        wide_name, wide = max(present, key=lambda pr: len(pr[1]["judges"]))
        ag = wide["robust"][s]["agreement"]
        L += ["", f"### 판정자 쌍 일치 — 구성 {wide_name}(판정자가 가장 많은 구성), 문항 {ag['n_items']}", "",
              "| 쌍 | 일치 % | Cohen κ |", "|---|---:|---:|"]
        for pair, blk in ag["pairwise"].items():
            L.append(f"| {pair.replace('__', ' × ')} | {blk['percent_agreement']} | {blk['cohen_kappa']} |")
        L += ["", "### 강건성 (구성별)", "",
              "| 구성 | Fleiss κ | 만장일치 % | O율차 %p [95% CI] | OR [95% CI] | MH OR(책) [95% CI] |",
              "|---|---:|---:|---|---|---|"]
        for p, r in present:
            rs = r["robust"][s]
            ov = rs["interval_estimates"]["o_vs_rest"]
            orr = ov["odds_ratio"]
            mh = rs["stratified"]["mh_book"]
            L.append(f"| {p} | {rs['agreement']['fleiss_kappa']} | {rs['agreement']['unanimous_pct']} | "
                     f"{ov['diff_pp']} {ov.get('diff_ci95_pp')} | {orr['or']} {orr['ci95']} | "
                     f"{(str(mh['or']) + ' ' + str(mh['ci95'])) if mh else '—'} |")
        L += ["", "### 합의 정의 민감도 (구성별 O율차 %p)", "", "| 구성 | 만장일치 | 과반 | 1표 이상 |", "|---|---:|---:|---:|"]
        for p, r in present:
            sen = r["robust"][s]["sensitivity"]
            L.append(f"| {p} | {sen['unanimous']['diff_pp']} | {sen['majority']['diff_pp']} | {sen['any']['diff_pp']} |")
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threshold", type=float, default=0.5, help="판정 모델 확률이 이 값 이상이면 O")
    ap.add_argument("--order", choices=list(rj.ORDERS), default="single",
                    help="읽을 판정 결과의 묶는 방식(run_jev_judgments --order)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260611, help="강건성 보고서와 같은 부트스트랩 시드")
    ap.add_argument("--allow-partial", action="store_true",
                    help="판정이 일부만 있는 섹션도 낸다 — 모든 판정자를 공통 문항으로 제한한다")
    ap.add_argument("--source", choices=["auto", "raw", "anon"], default="auto",
                    help="3모델 판정 CSV 소스(compute_final_stats와 같다). 공개 클론은 anon")
    a = ap.parse_args(argv)
    meta_run = {"threshold": a.threshold, "order": a.order, "source": a.source,
                "allow_partial": a.allow_partial}

    saved_source = cfs.SOURCE
    cfs.SOURCE = a.source
    loader = Loader(a.threshold, a.order)
    try:
        loader.install()
        results = OrderedDict()
        for name, judges in PANELS.items():
            print(f"[{name}] …")
            results[name] = run_panel(name, judges, loader, meta_run, a.boot, a.seed, a.allow_partial)
    finally:
        loader.restore()
        cfs.SOURCE = saved_source
    if loader.skipped_lines:
        print("잘린 JSONL 줄을 건너뛰었다:", loader.skipped_lines)
    if results.get("3models"):
        diffs = check_baseline(results["3models"])
        print("기준(3models) 정본 대조:", "PASS" if not diffs else "FAIL " + "; ".join(diffs))
        if diffs:
            return 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUMMARY.md").write_text(summary(results, meta_run), encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
