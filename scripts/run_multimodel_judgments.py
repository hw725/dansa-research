#!/usr/bin/env python3
"""
단사(斷辭) 전수조사 - 멀티모델 교차검증
=====================================================
3종 LLM에 동일 프롬프트를 병렬 호출하여 동일 데이터셋의 의미 자질을
교차 판정한다. 다른 한문 데이터셋에도 동일 프레임워크로 재사용 가능.

사용법:
  python scripts/run_multimodel_judgments.py
      모든 모델 × 모든 섹션 (기본 순서: section3 → section1 → section2)

  python scripts/run_multimodel_judgments.py --models gpt5mini,claude_sonnet
      모델 선택

  python scripts/run_multimodel_judgments.py --sections section3,section1
      섹션 선택

  python scripts/run_multimodel_judgments.py --ordering model-first
      모델별 전 섹션을 순차 처리 (기본은 section-first)

다른 데이터셋 적용:
  - DATA_FILE 변경
  - prepare_subsets()의 필터 조건 수정
  - SECTIONS dict의 prompt/subset 키 추가·수정
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy import stats
from tqdm import tqdm

load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / "env")

# ============================================================
# 모델 설정
# ============================================================
MODELS = {
    "gpt5mini": {
        "display": "GPT-5-mini",
        "model_id": "gpt-5-mini",
        "client_kwargs": {},
    },
    "gemini": {
        "display": "Gemini 2.5 Flash",
        "model_id": "google/gemini-2.5-flash",
        "client_kwargs": {
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "base_url": "https://openrouter.ai/api/v1",
        },
    },
    "claude_sonnet": {
        "display": "Claude Sonnet",
        "model_id": "~anthropic/claude-sonnet-latest",
        "client_kwargs": {
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "base_url": "https://openrouter.ai/api/v1",
        },
    },
}

BATCH_SIZE = 20
CONCURRENT_REQUESTS = 10
DATA_FILE = Path("data/sentence_normalized.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 동일 CSV에 여러 batch가 동시에 append하는 race condition 방지용 락
_csv_locks: dict[Path, asyncio.Lock] = {}


def _lock_for(path: Path) -> asyncio.Lock:
    if path not in _csv_locks:
        _csv_locks[path] = asyncio.Lock()
    return _csv_locks[path]


def get_client(model_key: str) -> AsyncOpenAI:
    cfg = MODELS[model_key]
    return AsyncOpenAI(**cfg["client_kwargs"])


# ============================================================
# 프롬프트
# ============================================================
def build_section1_prompt(translations: list[str]) -> str:
    texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(translations)])
    return f"""다음 번역문이 "가벼운 감탄이나 여운을 남기며 마무리하는" 뉘앙스인지 판단해주세요.

"가벼운 감탄·여운"이란:
- 무거운 논단이나 엄격한 결론이 아닌, 가벼운 어감의 마무리
- 정서적 고양이 동반됨

해당되면 O, 아니면 X로 답해주세요.
{texts}
각 문장에 대해 번호와 O/X만 답해주세요. 예: "1. O"
"""


def build_section2_decision_prompt(translations: list[str]) -> str:
    """夬絶之斷의 任圭直 정의("결정하여 끊는 말")를 표준국어대사전의
    ‘결정하다’("행동이나 태도를 분명하게 정하다") 풀이에 근거하여
    조작화한 프롬프트."""
    texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(translations)])
    return f"""다음 번역문에서 화자가 행동이나 태도를 분명하게 정하며 마무리하는지 판단해주세요.

"행동이나 태도를 분명하게 정하며 마무리"란:
- 화자가 입장·판단·방침을 분명하게 확정하는 마침
- 분명한 결론이나 판단을 내리며 끝맺는 진술

해당되면 O, 아니면 X로 답해주세요.
{texts}
각 문장에 대해 번호와 O/X만 답해주세요. 예: "1. O"
"""


def build_section3_prompt(translations: list[str]) -> str:
    texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(translations)])
    return f"""다음 번역문들이 "두루 진술하는" 내용인지 판단해주세요.

"두루 진술"이란:
- 특정 사건·인물이 아닌 일반론을 서술하는 것
- 개별 상황이 아닌 통론적 진술

해당되면 O, 아니면 X로 답해주세요.
{texts}
각 문장에 대해 번호와 O/X만 답해주세요. 예: "1. O"
"""


def parse_ox_response(result_text: str, count: int) -> list[bool]:
    results = []
    for i in range(1, count + 1):
        if f"{i}. O" in result_text or f"{i}.O" in result_text:
            results.append(True)
        else:
            results.append(False)
    return results


# ============================================================
# 비동기 LLM 호출
# ============================================================
async def call_llm(client: AsyncOpenAI, model_id: str, prompt: str) -> str:
    try:
        kwargs = {"model": model_id,
                  "messages": [{"role": "user", "content": prompt}]}
        if model_id.startswith("gpt"):
            pass  # no token limit — reasoning models consume tokens for thinking
        else:
            kwargs["max_tokens"] = 200
        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"  LLM 오류: {e}")
        return ""


async def append_to_csv_safe(df: pd.DataFrame, filepath: Path):
    """asyncio.Lock 으로 동시 쓰기 보호."""
    async with _lock_for(filepath):
        write_header = not filepath.exists()
        df.to_csv(filepath, mode='a', header=write_header,
                  index=False, encoding='utf-8')


async def process_batches(
    data_df: pd.DataFrame,
    client: AsyncOpenAI,
    model_id: str,
    build_prompt_fn,
    marker_type: str,
    section: str,
    csv_path: Path,
    desc: str,
):
    translations = data_df['번역문'].fillna('').tolist()
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    batches = []
    for i in range(0, len(translations), BATCH_SIZE):
        batch_trans = translations[i:i + BATCH_SIZE]
        batch_df = data_df.iloc[i:i + BATCH_SIZE].copy()
        batches.append((i, batch_trans, batch_df))

    async def process_one(batch_info):
        idx, batch_trans, batch_df = batch_info
        async with semaphore:
            prompt = build_prompt_fn(batch_trans)
            result_text = await call_llm(client, model_id, prompt)
            judgments = parse_ox_response(result_text, len(batch_trans))

            batch_df = batch_df.copy()
            batch_df['llm_judgment'] = judgments[:len(batch_df)]
            batch_df['marker_type'] = marker_type
            batch_df['analysis_section'] = section

            cols = ['book', '문단식별자', '문장식별자', '원문', '번역문',
                    'marker_raw', 'marker_normalized', 'dansa_category',
                    'llm_judgment', 'marker_type', 'analysis_section']
            cols = [c for c in cols if c in batch_df.columns]
            await append_to_csv_safe(batch_df[cols], csv_path)
            return len(batch_df)

    pbar = tqdm(total=len(batches), desc=desc)
    tasks = [process_one(b) for b in batches]
    for coro in asyncio.as_completed(tasks):
        await coro
        pbar.update(1)
    pbar.close()


# ============================================================
# 데이터 준비
# ============================================================
def prepare_subsets(df: pd.DataFrame) -> dict:
    """다른 데이터셋에 적용 시 이 함수의 필터 조건을 데이터에 맞게 수정."""
    roda = df[df['dansa_category'] == '游辭以斷'].copy()
    nira = df[df['dansa_category'] == '夬絶之斷'].copy()
    ra = df[df['dansa_category'] == '微絶之斷'].copy()

    # section1: 로다 전체 + 동수 라 대조군
    section1_control = ra.sample(n=len(roda), random_state=42).copy()

    # section2: 니라 vs 라 동수 매칭
    section2_n = min(len(nira), len(ra))
    section2_target = nira.sample(n=section2_n, random_state=42).copy()
    section2_control = ra.sample(n=section2_n, random_state=42).copy()

    # section3: 하나니라(汎論以斷 종결형) vs 동수 라 대조군
    hananira = df[
        (df['dansa_category'] == '汎論以斷') &
        (df['marker_normalized'] == '하나니라')
    ].copy()
    section3_control = ra.sample(n=len(hananira), random_state=99).copy()

    return {
        "section1_target": roda,
        "section1_control": section1_control,
        "section2_target": section2_target,
        "section2_control": section2_control,
        "section3_target": hananira,
        "section3_control": section3_control,
    }


# ============================================================
# 섹션 정의
# ============================================================
SECTIONS = {
    "section1": {
        "display": "Section 1: 游辭以斷 (감탄/여운)",
        "prompt": build_section1_prompt,
        "csv": "section1_judgments.csv",
        "target_subset": "section1_target",
        "control_subset": "section1_control",
        "target_label": "로다",
        "control_label": "라(대조군)",
    },
    "section2": {
        "display": "Section 2: 夬絶·微絶 (행동·태도를 분명하게 정함)",
        "prompt": build_section2_decision_prompt,
        "csv": "section2_decision_judgments.csv",
        "target_subset": "section2_target",
        "control_subset": "section2_control",
        "target_label": "니라",
        "control_label": "라",
    },
    "section3": {
        "display": "Section 3: 汎論以斷 (보편성)",
        "prompt": build_section3_prompt,
        "csv": "section3_judgments.csv",
        "target_subset": "section3_target",
        "control_subset": "section3_control",
        "target_label": "하나니라",
        "control_label": "라(대조군)",
    },
}

# 기본 실행 순서: 짧은 섹션 먼저 → 긴 섹션 마지막
# (중간 크래시 시 무거운 section2가 가장 적게 영향받음)
DEFAULT_SECTION_ORDER = ["section3", "section1", "section2"]

SUPPLEMENT_CSV_BY_SECTION = {
    "section1": "supplement_section1_judgments.csv",
    "section2": "supplement_section2_judgments.csv",
    "section3": "supplement_section3_judgments.csv",
}

LLM_MANIFEST_DIR = Path("data/llm_manifests")
BASE_MANIFEST_BY_SECTION = {
    "section1": "section1_base.csv",
    "section2": "section2_base.csv",
    "section3": "section3_base.csv",
}


def resolve_key_cols(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    if "book" not in cols or len(cols) < 3:
        raise KeyError("required key columns are missing")
    return ["book", cols[1], cols[2]]


def apply_llm_input_manifests(subsets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for section_key, manifest_name in BASE_MANIFEST_BY_SECTION.items():
        path = LLM_MANIFEST_DIR / manifest_name
        if not path.exists():
            continue
        section_cfg = SECTIONS[section_key]
        manifest = pd.read_csv(path)
        if "marker_type" not in manifest.columns:
            raise ValueError(f"manifest missing marker_type: {path}")
        manifest = manifest.drop(columns=["llm_judgment"], errors="ignore")
        subsets[section_cfg["target_subset"]] = manifest[
            manifest["marker_type"] == section_cfg["target_label"]
        ].copy()
        subsets[section_cfg["control_subset"]] = manifest[
            manifest["marker_type"] == section_cfg["control_label"]
        ].copy()
    return subsets


# ============================================================
# 단일 (모델 × 섹션) 실행
# ============================================================
async def run_one_section(model_key: str, section_key: str, subsets: dict):
    cfg = MODELS[model_key]
    display = cfg["display"]
    model_id = cfg["model_id"]
    model_dir = RESULTS_DIR / model_key
    model_dir.mkdir(exist_ok=True)

    section_cfg = SECTIONS[section_key]
    csv_path = model_dir / section_cfg["csv"]

    target = subsets[section_cfg["target_subset"]]
    control = subsets[section_cfg["control_subset"]]

    _KEY_COLS = resolve_key_cols(target)

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        ex_t = existing[existing['marker_type'] == section_cfg["target_label"]]
        ex_c = existing[existing['marker_type'] == section_cfg["control_label"]]
        done_target_keys = set(ex_t[_KEY_COLS].apply(tuple, axis=1))
        done_control_keys = set(ex_c[_KEY_COLS].apply(tuple, axis=1))
    else:
        done_target_keys = set()
        done_control_keys = set()

    supplement_csv = SUPPLEMENT_CSV_BY_SECTION.get(section_key)
    if supplement_csv:
        supplement_path = model_dir / supplement_csv
        if supplement_path.exists():
            supplement = pd.read_csv(supplement_path)
            if all(col in supplement.columns for col in [*_KEY_COLS, "marker_type"]):
                sup_t = supplement[
                    supplement["marker_type"] == section_cfg["target_label"]
                ]
                sup_c = supplement[
                    supplement["marker_type"] == section_cfg["control_label"]
                ]
                done_target_keys.update(set(sup_t[_KEY_COLS].apply(tuple, axis=1)))
                done_control_keys.update(set(sup_c[_KEY_COLS].apply(tuple, axis=1)))

    target_todo = target[~target[_KEY_COLS].apply(tuple, axis=1).isin(done_target_keys)]
    control_todo = control[~control[_KEY_COLS].apply(tuple, axis=1).isin(done_control_keys)]

    n_done_t = len(target) - len(target_todo)
    n_done_c = len(control) - len(control_todo)

    print(f"\n[{section_key}] {display}: "
          f"target {len(target_todo):,} todo ({n_done_t} done), "
          f"control {len(control_todo):,} todo ({n_done_c} done)")

    if len(target_todo) == 0 and len(control_todo) == 0:
        print(f"  → 전부 완료, 스킵")
        return

    client = get_client(model_key)
    if len(target_todo) > 0:
        await process_batches(target_todo, client, model_id, section_cfg["prompt"],
                              section_cfg["target_label"], section_key, csv_path,
                              f"[{display}] {section_key} {section_cfg['target_label']}")
    if len(control_todo) > 0:
        await process_batches(control_todo, client, model_id, section_cfg["prompt"],
                              section_cfg["control_label"], section_key, csv_path,
                              f"[{display}] {section_key} {section_cfg['control_label']}")


# ============================================================
# 통계
# ============================================================
def _compute_section_stats(csv_path, target_label, control_label):
    df = pd.read_csv(csv_path)
    tgt = df[df['marker_type'] == target_label]
    ctrl = df[df['marker_type'] == control_label]
    tp = int(tgt['llm_judgment'].sum())
    cp = int(ctrl['llm_judgment'].sum())
    ct = [[tp, len(tgt) - tp], [cp, len(ctrl) - cp]]
    chi2, pv = stats.chi2_contingency(ct)[:2]
    return {
        "n_target": len(tgt), "n_control": len(ctrl),
        "target_positive": tp, "control_positive": cp,
        "target_rate": round(tp / len(tgt) * 100, 2) if len(tgt) else 0,
        "control_rate": round(cp / len(ctrl) * 100, 2) if len(ctrl) else 0,
        "chi2": round(chi2, 2), "p_value": float(f"{pv:.3e}"),
        "significant": bool(pv < 0.05),
    }


def compute_all_stats(model_keys: list[str], section_keys: list[str]) -> dict:
    out = RESULTS_DIR / "intermediate_multimodel_stats.json"
    if out.exists():
        with open(out, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        all_stats = payload.get("sections", {})
    else:
        all_stats = {}

    for mk in model_keys:
        cfg = MODELS[mk]
        if mk not in all_stats:
            all_stats[mk] = {"display": cfg["display"], "sections": {}}
        for lk in section_keys:
            section_cfg = SECTIONS[lk]
            csv_path = RESULTS_DIR / mk / section_cfg["csv"]
            if csv_path.exists():
                s = _compute_section_stats(
                    csv_path,
                    section_cfg["target_label"],
                    section_cfg["control_label"],
                )
                all_stats[mk]["sections"][lk] = s
                print(f"[{cfg['display']}] {lk}: "
                      f"{section_cfg['target_label']} {s['target_rate']}% vs "
                      f"{section_cfg['control_label']} {s['control_rate']}% "
                      f"(p={s['p_value']})")

    payload = {
        "survey_completed_at": __import__("datetime").date.today().isoformat(),
        "manuscript_version": "v3.1_cleaned_base_before_control_supplements",
        "gemini_via": "OpenRouter API",
        "sections": all_stats,
    }
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out}")
    return all_stats


# ============================================================
# 메인
# ============================================================
async def main_async(model_keys: list[str], section_keys: list[str], ordering: str):
    print("=" * 60)
    print("단사(斷辭) 전수조사 - 멀티모델 교차검증")
    print(f"모델: {', '.join(MODELS[k]['display'] for k in model_keys)}")
    print(f"섹션: {', '.join(section_keys)}  (순서: {ordering})")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"\n데이터 로드: {DATA_FILE}  ({len(df):,}행)")
    subsets = apply_llm_input_manifests(prepare_subsets(df))

    if ordering == "section-first":
        for lk in section_keys:
            print(f"\n{'=' * 60}")
            print(f"{SECTIONS[lk]['display']} — 모든 모델 (병렬)")
            print(f"{'=' * 60}")
            await asyncio.gather(*[run_one_section(mk, lk, subsets) for mk in model_keys])
    elif ordering == "model-first":
        for mk in model_keys:
            print(f"\n{'=' * 60}")
            print(f"{MODELS[mk]['display']} — 모든 섹션")
            print(f"{'=' * 60}")
            for lk in section_keys:
                await run_one_section(mk, lk, subsets)
    else:
        raise ValueError(f"unknown ordering: {ordering}")

    print(f"\n{'=' * 60}")
    print("통계 계산")
    print(f"{'=' * 60}")
    compute_all_stats(model_keys, section_keys)
    print("\n완료!")


def main():
    parser = argparse.ArgumentParser(description="단사 멀티모델 교차검증")
    parser.add_argument("--models", default=",".join(MODELS.keys()),
                        help=f"콤마 구분 모델 리스트 (기본: 전체). 가능: {','.join(MODELS.keys())}")
    parser.add_argument("--sections", default=",".join(DEFAULT_SECTION_ORDER),
                        help=f"콤마 구분 섹션 리스트 (기본: {','.join(DEFAULT_SECTION_ORDER)}). 가능: {','.join(SECTIONS.keys())}")
    parser.add_argument("--ordering", choices=["section-first", "model-first"],
                        default="section-first",
                        help="실행 순서. section-first(기본): 모든 모델의 같은 섹션을 순차로. model-first: 모델별 전 섹션을 순차로.")
    parser.add_argument("--model", help=argparse.SUPPRESS)  # 구버전 호환
    args = parser.parse_args()

    if args.model:  # 구버전 호환
        model_keys = [args.model]
    else:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    section_keys = [lk.strip() for lk in args.sections.split(",") if lk.strip()]

    for mk in model_keys:
        if mk not in MODELS:
            sys.exit(f"unknown model: {mk}")
    for lk in section_keys:
        if lk not in SECTIONS:
            sys.exit(f"unknown section: {lk}")

    asyncio.run(main_async(model_keys, section_keys, args.ordering))


if __name__ == "__main__":
    main()
