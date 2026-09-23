#!/usr/bin/env python3
"""Jev 판정을 3모델 판정에 대 본다(호출 0건).

무엇을 재는가(섹션마다):
    ① 순위: Jev 확률이 3모델 표수(0~3)와 같은 방향인가 — Spearman ρ, 그리고 만장일치 O 대
       만장일치 X를 가르는 AUC(0.5 = 동전, 1.0 = 완벽).
    ② 문턱별 일치: p ≥ t를 O로 볼 때 다수결(표 ≥ 2)과의 Cohen κ, 모델별 κ. 3모델끼리의 κ를
       같은 표에 두어야 «Jev가 한 명의 판정자로 쓸 만한가»가 읽힌다 — 절대값만으로는 모른다.
    ③ 효과 재현: target(표지 있음) 대 control의 O 비율 차와 2×2 χ²·φ. 3모델 합의가 보인
       차이를 독립 판정자가 같은 방향으로 내는가가 논문 주장에 닿는 질문이다.
    ④ flip 대조군: 반대 틀 질문의 확률 q와 짝지어 p + q가 1 근처인가, r(p, q)가 음인가.
       둘 다 ≥ 0.5(모순)인 비율이 «예로 기우는» 정도다.

입력: results/jev/{section}_{pos,neg}.jsonl, 3모델 판정 CSV(raw 없으면 익명본).
출력: results/jev/jev_agreement.json, 표준출력에 마크다운 표.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_final_stats as cfs  # noqa: E402
import run_jev_judgments as rj  # noqa: E402

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
OUT = rj.OUT_DIR / "jev_agreement.json"
JEV = rj.jev_client.TYPESAFE_MODEL


def out_json(model: str) -> Path:
    return OUT if model == JEV else rj.OUT_DIR / f"jev_agreement.{model}.json"


def read_probs(section: str, mode: str, model: str = JEV) -> dict[tuple, float]:
    path = rj.out_path(section, mode, model)
    out: dict[tuple, float] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("noul") is not None:
                    out[tuple(rec["key"])] = float(rec["noul"])
    return out


def ranks(xs: list[float]) -> list[float]:
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if not va or not vb:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney U / (n1·n2). pos가 neg보다 높을 확률(동점 ½)."""
    if not pos or not neg:
        return None
    r = ranks(pos + neg)
    u = sum(r[: len(pos)]) - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def r3(v):
    return None if v is None else round(v, 3)


def effect(flags: list[bool], is_target: list[bool]) -> dict:
    t = [f for f, it in zip(flags, is_target) if it]
    c = [f for f, it in zip(flags, is_target) if not it]
    a, b = sum(t), len(t) - sum(t)
    cc, d = sum(c), len(c) - sum(c)
    chi2, p = cfs.chi_square_2x2(a, b, cc, d)
    n = a + b + cc + d
    return {"target_O_pct": round(100 * a / len(t), 1) if t else None,
            "control_O_pct": round(100 * cc / len(c), 1) if c else None,
            "chi2": round(chi2, 2), "p": p, "phi": r3(math.sqrt(chi2 / n)) if n else None}


def score_section(section: str, model: str = JEV) -> dict | None:
    probs = read_probs(section, "pos", model)
    if not probs:
        return None
    other = {} if model == JEV else read_probs(section, "pos", JEV)
    items = [it for it in rj._load(section, need_text=False) if tuple(it["key"]) in probs]
    p = [probs[tuple(it["key"])] for it in items]
    votes = [it["votes"] for it in items]
    is_t = [it["is_target"] for it in items]
    majority = [v >= 2 for v in votes]
    models = list(cfs.MODELS)

    rho = pearson(ranks(p), ranks([float(v) for v in votes]))
    res = {
        "n": len(items),
        "n_target": sum(is_t),
        "n_control": len(items) - sum(is_t),
        "spearman_votes": r3(rho),
        "auc_unanimous": r3(auc([x for x, v in zip(p, votes) if v == 3],
                                [x for x, v in zip(p, votes) if v == 0])),
        "n_unanimous_O": sum(v == 3 for v in votes),
        "n_unanimous_X": sum(v == 0 for v in votes),
        "auc_target_vs_control": r3(auc([x for x, t in zip(p, is_t) if t],
                                        [x for x, t in zip(p, is_t) if not t])),
        "reference": {
            "consensus_unanimous": effect([v == 3 for v in votes], is_t),
            "majority": effect(majority, is_t),
            **{m: effect([it["per_model"][m] for it in items], is_t) for m in models},
            "kappa_between_models": {
                f"{a}~{b}": r3(kappa([it["per_model"][a] for it in items],
                                     [it["per_model"][b] for it in items]))
                for i, a in enumerate(models) for b in models[i + 1:]
            },
        },
        "thresholds": {},
    }
    for t in THRESHOLDS:
        jo = [x >= t for x in p]
        res["thresholds"][str(t)] = {
            "jev": effect(jo, is_t),
            "kappa_majority": r3(kappa(jo, majority)),
            **{f"kappa_{m}": r3(kappa(jo, [it["per_model"][m] for it in items])) for m in models},
        }

    # 다른 System One 모델이면 진짜 Jev와도 대 본다(같은 문항만)
    shared = [(probs[tuple(it["key"])], other[tuple(it["key"])], it) for it in items
              if tuple(it["key"]) in other]
    if shared:
        res["vs_jev"] = {
            "n": len(shared),
            "pearson": r3(pearson([a for a, _, _ in shared], [b for _, b, _ in shared])),
            "spearman": r3(pearson(ranks([a for a, _, _ in shared]), ranks([b for _, b, _ in shared]))),
            "kappa_at_0.5": r3(kappa([a >= 0.5 for a, _, _ in shared], [b >= 0.5 for _, b, _ in shared])),
            "jev_auc_unanimous_same_items": r3(auc([b for _, b, it in shared if it["votes"] == 3],
                                                   [b for _, b, it in shared if it["votes"] == 0])),
        }

    neg = read_probs(section, "neg", model)
    pairs = [(probs[k], neg[k]) for k in probs if k in neg]
    if pairs:
        s = [a + b for a, b in pairs]
        res["flip"] = {
            "n": len(pairs),
            "mean_p_plus_q": r3(sum(s) / len(s)),
            "mean_abs_dev_from_1": r3(sum(abs(x - 1) for x in s) / len(s)),
            "pearson_p_q": r3(pearson([a for a, _ in pairs], [b for _, b in pairs])),
            "both_yes_pct": round(100 * sum(a >= 0.5 and b >= 0.5 for a, b in pairs) / len(pairs), 1),
            "both_no_pct": round(100 * sum(a < 0.5 and b < 0.5 for a, b in pairs) / len(pairs), 1),
        }
    return res


def to_markdown(all_res: dict) -> str:
    out = []
    for s, r in all_res.items():
        ref = r["reference"]
        out.append(f"## {s} {cfs.SECTIONS[s]['label']} — n={r['n']} (target {r['n_target']} · control {r['n_control']})\n")
        out.append(f"- Spearman ρ(Jev 확률, 3모델 표수) = {r['spearman_votes']}")
        out.append(f"- AUC 만장일치 O({r['n_unanimous_O']}) 대 X({r['n_unanimous_X']}) = {r['auc_unanimous']}")
        out.append(f"- AUC target 대 control = {r['auc_target_vs_control']}")
        out.append(f"- 3모델끼리 κ: " + ", ".join(f"{k} {v}" for k, v in ref["kappa_between_models"].items()))
        out.append("")
        out.append("| 판정자 | target O% | control O% | χ² | φ | κ(다수결) |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for name in ["consensus_unanimous", "majority", *cfs.MODELS]:
            e = ref[name]
            out.append(f"| {name} | {e['target_O_pct']} | {e['control_O_pct']} | {e['chi2']} | {e['phi']} | |")
        for t, e in r["thresholds"].items():
            j = e["jev"]
            out.append(f"| Jev ≥ {t} | {j['target_O_pct']} | {j['control_O_pct']} | {j['chi2']} | {j['phi']} | {e['kappa_majority']} |")
        if "flip" in r:
            f = r["flip"]
            out.append("")
            out.append(f"flip 대조군 n={f['n']}: 평균 p+q = {f['mean_p_plus_q']}, |p+q−1| = {f['mean_abs_dev_from_1']}, "
                       f"r(p,q) = {f['pearson_p_q']}, 둘 다 예 {f['both_yes_pct']}% · 둘 다 아니오 {f['both_no_pct']}%")
        if "vs_jev" in r:
            v = r["vs_jev"]
            out.append(f"\n진짜 Jev와 n={v['n']}: Pearson {v['pearson']} · Spearman {v['spearman']} · "
                       f"κ(≥0.5) {v['kappa_at_0.5']} · 같은 문항 Jev AUC {v['jev_auc_unanimous_same_items']}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(rj.JUDGES), default=JEV)
    model = ap.parse_args(argv).model
    all_res = {}
    for s in cfs.SECTIONS:
        r = score_section(s, model)
        if r:
            all_res[s] = r
    if not all_res:
        print(f"채점할 {model} 결과가 없습니다 — 먼저 run_jev_judgments.py run.", file=sys.stderr)
        return 1
    OUT = out_json(model)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(to_markdown(all_res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
