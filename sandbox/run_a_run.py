"""run_a_run.py — 로컬 입력으로 LLM 판정을 다시 수행한다.

동결 기준과 동일한 프롬프트·표본(LLM manifest)·O/X 파서를 그대로 import 해 쓴다.
차이는 호출 경로(동결 실행은 gemini·claude를 OpenRouter 경유로 호출했고,
샌드박스는 벤더 직결 — vendor_clients.py 참조)와 비결정성에서 온다.

출력은 쓰기 가능한 임시 공간(A_RUN_DIR, 기본 /tmp/a_run)에 쓴다. 키 컬럼과
marker_type·llm_judgment 만 쓰고 번역문·원문은 쓰지 않는다.
"""
from __future__ import annotations

import asyncio
import csv
import os
import sys
from pathlib import Path

import pandas as pd

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "scripts"))
sys.path.insert(0, str(APP / "sandbox"))

import run_multimodel_judgments as rmj  # noqa: E402  (동결 기준과 동일한 방법 재사용)
from vendor_clients import SANDBOX_MODELS, available_models, call_vendor, get_client  # noqa: E402

BATCH = 20
CONCURRENT = 10
A_RUN_DIR = Path(os.environ.get("A_RUN_DIR", "/tmp/a_run"))


async def _judge_subset(model_key: str, section_key: str, subset: pd.DataFrame,
                        marker_label: str, out_rows: list[dict]) -> None:
    cfg = rmj.SECTIONS[section_key]
    prompt_fn = cfg["prompt"]
    key_cols = rmj.resolve_key_cols(subset)
    translations = subset["번역문"].fillna("").astype(str).tolist()
    meta = subset[key_cols].values.tolist()
    results: list[bool | None] = [None] * len(translations)

    client = get_client(model_key)
    sem = asyncio.Semaphore(CONCURRENT)

    async def one(start: int) -> None:
        batch = translations[start:start + BATCH]
        async with sem:
            text = await call_vendor(client, model_key, prompt_fn(batch))
        parsed = rmj.parse_ox_response(text, len(batch))
        for j, val in enumerate(parsed):
            results[start + j] = val

    await asyncio.gather(*[one(i) for i in range(0, len(translations), BATCH)])

    for key_vals, val in zip(meta, results):
        row = dict(zip(key_cols, key_vals))
        row["marker_type"] = marker_label
        row["llm_judgment"] = bool(val)
        row["analysis_section"] = section_key
        out_rows.append(row)


def _write(model_key: str, section_key: str, rows: list[dict]) -> Path:
    cfg = rmj.SECTIONS[section_key]
    dest = A_RUN_DIR / model_key / cfg["csv"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return dest
    cols = [c for c in ("book", "문단식별자", "문장식별자",
                        "marker_type", "llm_judgment", "analysis_section")
            if c in rows[0]]
    with open(dest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    return dest


async def main_async() -> int:
    if not rmj.DATA_FILE.exists():
        print(f"[rerun] corpus 없음: {rmj.DATA_FILE} (마운트 확인)")
        return 1

    models = available_models()
    if not models:
        print("[rerun] 사용할 키 없음 — OPENAI/ANTHROPIC/GEMINI 중 최소 1개 필요")
        return 1
    print(f"[rerun] 모델: {', '.join(SANDBOX_MODELS[m]['display'] for m in models)}")

    df = pd.read_csv(rmj.DATA_FILE)
    subsets = rmj.apply_llm_input_manifests(rmj.prepare_subsets(df))
    print(f"[rerun] corpus {len(df):,}행 · LLM manifest 표본 적용")

    for section_key in ("section1", "section2", "section3"):
        cfg = rmj.SECTIONS[section_key]
        target = subsets[cfg["target_subset"]]
        control = subsets[cfg["control_subset"]]
        print(f"\n[rerun] {section_key}  target {len(target):,} · control {len(control):,}")
        for model_key in models:
            rows: list[dict] = []
            await _judge_subset(model_key, section_key, target, cfg["target_label"], rows)
            await _judge_subset(model_key, section_key, control, cfg["control_label"], rows)
            dest = _write(model_key, section_key, rows)
            pos = sum(1 for r in rows if r["llm_judgment"])
            print(f"   {SANDBOX_MODELS[model_key]['display']:>16}: "
                  f"{len(rows):,}건 판정 (O={pos}) → {dest}")

    print(f"\n[rerun] 완료 → {A_RUN_DIR}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
