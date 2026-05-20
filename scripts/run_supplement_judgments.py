"""
보충 대조군 3모델 판정 파이프라인
- supplement_section1_control_12.csv  (12건)
- supplement_section2_control_465.csv (465건)
- supplement_section3_control_30.csv (30건)

모델: GPT-5-mini, Gemini 2.5 Flash, Claude Sonnet 4.6
"""
import argparse
import asyncio
import os
import re
import sys
import json
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

# ============================================================
# 환경 설정 — OPENROUTER_API_KEY를 backend-44/.env에서 로드
# ============================================================
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = _SCRIPT_DIR / "env"
if ENV_FILE.exists():
    with open(ENV_FILE, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
BATCH_SIZE = 20

SUPPLEMENT_FILES = {
    "section1":  DATA_DIR / "supplement_section1_control_12.csv",
    "section2":  DATA_DIR / "supplement_section2_control_465.csv",
    "section3": DATA_DIR / "supplement_section3_control_30.csv",
}

# ============================================================
# 프롬프트 (run_multimodel_judgments.py에서 복사)
# ============================================================
def build_section1_prompt(translations):
    texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(translations)])
    return f"""다음 번역문이 "가벼운 감탄이나 여운을 남기며 마무리하는" 뉘앙스인지 판단해주세요.

"가벼운 감탄·여운"이란:
- 무거운 논단이나 엄격한 결론이 아닌, 가벼운 어감의 마무리
- 정서적 고양이 동반됨

해당되면 O, 아니면 X로 답해주세요.
{texts}
각 문장에 대해 번호와 O/X만 답해주세요. 예: "1. O"
"""


def build_section2_prompt(translations):
    texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(translations)])
    return f"""다음 번역문에서 화자가 행동이나 태도를 분명하게 정하며 마무리하는지 판단해주세요.

"행동이나 태도를 분명하게 정하며 마무리"란:
- 화자가 입장·판단·방침을 분명하게 확정하는 마침
- 분명한 결론이나 판단을 내리며 끝맺는 진술

해당되면 O, 아니면 X로 답해주세요.
{texts}
각 문장에 대해 번호와 O/X만 답해주세요. 예: "1. O"
"""


def build_section3_prompt(translations):
    texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(translations)])
    return f"""다음 번역문들이 "두루 진술하는" 내용인지 판단해주세요.

"두루 진술"이란:
- 특정 사건·인물이 아닌 일반론을 서술하는 것
- 개별 상황이 아닌 통론적 진술

해당되면 O, 아니면 X로 답해주세요.
{texts}
각 문장에 대해 번호와 O/X만 답해주세요. 예: "1. O"
"""


PROMPTS = {
    "section1":  build_section1_prompt,
    "section2":  build_section2_prompt,
    "section3": build_section3_prompt,
}

# ============================================================
# 모델 설정 — env 로드 후 구성
# ============================================================
def _build_models():
    ork = os.environ.get("OPENROUTER_API_KEY", "")
    return {
        "gpt5mini": {
            "display": "GPT-5-mini",
            "model_id": "gpt-5-mini",
            "client_kwargs": {},
        },
        "gemini": {
            "display": "Gemini 2.5 Flash",
            "model_id": "google/gemini-2.5-flash",
            "client_kwargs": {"api_key": ork, "base_url": "https://openrouter.ai/api/v1"},
        },
        "claude_sonnet": {
            "display": "Claude Sonnet",
            "model_id": "~anthropic/claude-sonnet-latest",
            "client_kwargs": {"api_key": ork, "base_url": "https://openrouter.ai/api/v1"},
        },
    }

MODELS = _build_models()


# ============================================================
# 판정 함수
# ============================================================
def parse_ox(result_text, n):
    """O/X 응답 파싱. 번호+O → True, else → False."""
    results = []
    for i in range(1, n + 1):
        if f"{i}. O" in result_text or f"{i}.O" in result_text:
            results.append(True)
        else:
            results.append(False)
    return results


async def judge_batch(model_key, prompt, n, retries=3):
    """OpenAI-compatible API로 판정 (GPT, Gemini, Claude 모두 동일 인터페이스)."""
    cfg = MODELS[model_key]
    client = AsyncOpenAI(**cfg["client_kwargs"])
    model_id = cfg["model_id"]

    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            return parse_ox(resp.choices[0].message.content, n)
        except Exception as e:
            print(f"  [{model_key}] attempt {attempt+1} error: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
    return [False] * n


# ============================================================
# 메인 파이프라인
# ============================================================
async def run_model_section(model_key, section_key, df):
    """하나의 (모델, 섹션) 조합에 대해 판정 실행."""
    prompt_fn = PROMPTS[section_key]
    translations = df["번역문"].tolist()
    n_total = len(translations)
    judgments = [None] * n_total

    print(f"\n{'='*60}")
    print(f"[{model_key}] × [{section_key}]: {n_total}건")
    print(f"{'='*60}")

    # 배치 분할
    batches = []
    for i in range(0, n_total, BATCH_SIZE):
        batch_trans = translations[i:i+BATCH_SIZE]
        batches.append((i, batch_trans))

    for batch_idx, (start, batch_trans) in enumerate(batches):
        prompt = prompt_fn(batch_trans)
        n_batch = len(batch_trans)

        results = await judge_batch(model_key, prompt, n_batch)

        for j, r in enumerate(results):
            judgments[start + j] = r

        o_count = sum(1 for r in results if r)
        print(f"  batch {batch_idx+1}/{len(batches)}: {o_count}/{n_batch} O")

    df = df.copy()
    df["llm_judgment"] = judgments
    return df


async def main(force: bool = False, dry_run: bool = False):
    start_time = time.time()
    models = ["gpt5mini", "gemini", "claude_sonnet"]
    sections = ["section1", "section2", "section3"]
    all_results = {}

    for section in sections:
        csv_path = SUPPLEMENT_FILES[section]
        if not csv_path.exists():
            print(f"SKIP: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8")
        print(f"\nLoaded {section}: {len(df)} rows from {csv_path.name}")

        for model in models:
            out_dir = RESULTS_DIR / model
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"supplement_{section.lower()}_judgments.csv"
            if out_path.exists() and not force:
                print(f"  SKIP existing: {out_path}")
                result_df = pd.read_csv(out_path, encoding="utf-8")
                all_results[(model, section)] = result_df
                continue
            if dry_run:
                print(f"  DRY RUN would write: {out_path}")
                continue

            result_df = await run_model_section(model, section, df)
            all_results[(model, section)] = result_df
            result_df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"  saved: {out_path}")

    # 요약
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"전체 완료: {elapsed:.1f}초")
    print(f"{'='*60}")

    for (model, section), df in all_results.items():
        o = df["llm_judgment"].sum()
        n = len(df)
        print(f"  {model} × {section}: {o}/{n} O ({o/n*100:.1f}%)")

    # 결과 요약 JSON
    summary = {}
    for (model, section), df in all_results.items():
        key = f"{model}_{section}"
        summary[key] = {
            "n": len(df),
            "positive": int(df["llm_judgment"].sum()),
            "rate": round(df["llm_judgment"].mean() * 100, 2),
        }
    if dry_run:
        print("\nDRY RUN summary not written")
        return
    summary_path = RESULTS_DIR / "supplement_judgment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="supplement judgment runner")
    parser.add_argument("--force", action="store_true", help="rerun and overwrite existing supplement judgments")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and output paths without API calls")
    args = parser.parse_args()
    asyncio.run(main(force=args.force, dry_run=args.dry_run))
