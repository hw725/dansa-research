#!/usr/bin/env python3
"""Non-destructive preflight for first-pass LLM judgment pipelines.

This script checks the path from normalized/raw-derived inputs to the first
LLM judgment outputs without calling external APIs or overwriting results.
It intentionally uses import stubs for network clients so prompt building,
resume accounting, parser behavior, and batch assembly can be verified offline.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - diagnostic path
    raise SystemExit("pandas is required for this preflight") from exc

REPO = Path(__file__).resolve().parent.parent
LOG_PATH = REPO / "logs" / "preflight_llm_pipeline.jsonl"

RUNTIME_DEPS = [
    "pandas",
    "openai",
    "scipy",
    "tqdm",
    "dotenv",
    "regex",
    "lxml",
    "sklearn",
]

RAW_SOURCE_CANDIDATES = [
    "primary_data",
    "datasets/sentence_full.csv",
    "datasets/phrase_full.csv",
    "datasets/sentence/train.csv",
    "datasets/sentence/val.csv",
    "datasets/sentence/test.csv",
    "datasets/phrase/train.csv",
    "datasets/phrase/val.csv",
    "datasets/phrase/test.csv",
]


def _install_import_stubs() -> None:
    """Install lightweight modules needed for offline imports."""

    if importlib.util.find_spec("openai") is None:
        openai = types.ModuleType("openai")

        class AsyncOpenAI:  # noqa: D401 - simple stub
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        openai.AsyncOpenAI = AsyncOpenAI
        sys.modules["openai"] = openai

    if importlib.util.find_spec("dotenv") is None:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: False
        sys.modules["dotenv"] = dotenv

    if importlib.util.find_spec("scipy") is None:
        scipy = types.ModuleType("scipy")
        stats = types.ModuleType("scipy.stats")

        def chi2_contingency(table: Any) -> tuple[float, float, int, list[list[float]]]:
            rows = len(table)
            cols = len(table[0]) if rows else 0
            return 0.0, 1.0, max((rows - 1) * (cols - 1), 0), []

        stats.chi2_contingency = chi2_contingency
        scipy.stats = stats
        sys.modules["scipy"] = scipy
        sys.modules["scipy.stats"] = stats

    if importlib.util.find_spec("tqdm") is None:
        tqdm_mod = types.ModuleType("tqdm")

        class TqdmStub:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.total = kwargs.get("total")

            def update(self, value: int = 1) -> None:
                return None

            def close(self) -> None:
                return None

        tqdm_mod.tqdm = TqdmStub
        sys.modules["tqdm"] = tqdm_mod


def _missing_runtime_deps() -> list[str]:
    missing: list[str] = []
    for dep in RUNTIME_DEPS:
        if importlib.util.find_spec(dep) is None:
            missing.append(dep)
    return missing


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _file_state(path: Path) -> dict[str, Any]:
    return {"path": _rel(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}


def _load_modules() -> tuple[Any, Any]:
    _install_import_stubs()
    sys.path.insert(0, str(REPO / "scripts"))
    import run_multimodel_judgments as multimodel
    import run_supplement_judgments as supplement

    return multimodel, supplement


def _count_existing(path: Path, target_label: str, control_label: str) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0, "target_done": 0, "control_done": 0}
    df = pd.read_csv(path)
    return {
        "exists": True,
        "rows": int(len(df)),
        "target_done": int((df["marker_type"] == target_label).sum()) if "marker_type" in df else None,
        "control_done": int((df["marker_type"] == control_label).sum()) if "marker_type" in df else None,
    }


def _key_set(df: pd.DataFrame, key_cols: list[str]) -> set[tuple[Any, ...]]:
    return set(df[key_cols].apply(tuple, axis=1))


def _resolve_key_cols(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    if "book" not in cols or len(cols) < 3:
        raise KeyError("required key columns are missing")
    return ["book", cols[1], cols[2]]


def _mock_ox(count: int) -> str:
    return "\n".join(f"{idx}. {'O' if idx % 2 else 'X'}" for idx in range(1, count + 1))


async def _check_multimodel_mock(multimodel: Any, subsets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    collected: list[pd.DataFrame] = []

    async def fake_call_llm(client: Any, model_id: str, prompt: str) -> str:
        count = sum(1 for line in prompt.splitlines() if line[:1].isdigit() and "." in line[:4])
        return _mock_ox(max(count, 1))

    async def fake_append(df: pd.DataFrame, filepath: Path) -> None:
        collected.append(df.copy())

    original_call = multimodel.call_llm
    original_append = multimodel.append_to_csv_safe
    multimodel.call_llm = fake_call_llm
    multimodel.append_to_csv_safe = fake_append
    try:
        checks: dict[str, Any] = {}
        for section_key, cfg in multimodel.SECTIONS.items():
            rows = subsets[cfg["target_subset"]].head(3)
            before = len(collected)
            await multimodel.process_batches(
                rows,
                object(),
                "mock-model",
                cfg["prompt"],
                cfg["target_label"],
                section_key,
                Path("mock.csv"),
                f"mock {section_key}",
            )
            frames = collected[before:]
            got_rows = sum(len(frame) for frame in frames)
            checks[section_key] = {
                "mock_rows": int(got_rows),
                "has_judgment": bool(frames and "llm_judgment" in frames[0].columns),
                "has_marker_type": bool(frames and "marker_type" in frames[0].columns),
                "has_analysis_section": bool(frames and "analysis_section" in frames[0].columns),
            }
        return checks
    finally:
        multimodel.call_llm = original_call
        multimodel.append_to_csv_safe = original_append


async def _check_supplement_mock(supplement: Any) -> dict[str, Any]:
    async def fake_judge_batch(model_key: str, prompt: str, n: int, retries: int = 3) -> list[bool]:
        return [(idx % 2) == 0 for idx in range(n)]

    original_judge = supplement.judge_batch
    supplement.judge_batch = fake_judge_batch
    try:
        checks: dict[str, Any] = {}
        for section_key, path in supplement.SUPPLEMENT_FILES.items():
            if not path.exists():
                checks[section_key] = {"input_exists": False}
                continue
            df = pd.read_csv(path, encoding="utf-8")
            result = await supplement.run_model_section("gpt5mini", section_key, df.head(3))
            checks[section_key] = {
                "input_exists": True,
                "input_rows": int(len(df)),
                "mock_rows": int(len(result)),
                "has_judgment": "llm_judgment" in result.columns,
            }
        return checks
    finally:
        supplement.judge_batch = original_judge


def _check_prompt_parsers(multimodel: Any, supplement: Any) -> dict[str, Any]:
    mm: dict[str, Any] = {}
    for section_key, cfg in multimodel.SECTIONS.items():
        prompt = cfg["prompt"](["alpha", "beta", "gamma"])
        parsed = multimodel.parse_ox_response("1. O\n2. X\n3. O", 3)
        mm[section_key] = {"prompt_chars": len(prompt), "parser_ok": parsed == [True, False, True]}

    sp: dict[str, Any] = {}
    for section_key, prompt_fn in supplement.PROMPTS.items():
        prompt = prompt_fn(["alpha", "beta", "gamma"])
        parsed = supplement.parse_ox("1. O\n2. X\n3. O", 3)
        sp[section_key] = {"prompt_chars": len(prompt), "parser_ok": parsed == [True, False, True]}

    return {"multimodel": mm, "supplement": sp}


def _write_log(payload: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


async def run_preflight(strict: bool = False) -> tuple[int, dict[str, Any]]:
    missing_before_stubs = _missing_runtime_deps()
    multimodel, supplement = _load_modules()

    sentence_path = REPO / multimodel.DATA_FILE
    sentence_exists = sentence_path.exists()
    sentence_rows = 0
    base_counts: dict[str, Any] = {}
    mock_multimodel: dict[str, Any] = {}

    if sentence_exists:
        df = pd.read_csv(sentence_path)
        sentence_rows = int(len(df))
        subsets = multimodel.apply_llm_input_manifests(multimodel.prepare_subsets(df))
        for section_key, cfg in multimodel.SECTIONS.items():
            target = subsets[cfg["target_subset"]]
            control = subsets[cfg["control_subset"]]
            key_cols = _resolve_key_cols(target)
            target_keys = _key_set(target, key_cols)
            control_keys = _key_set(control, key_cols)
            section_counts: dict[str, Any] = {
                "target_expected": int(len(target)),
                "control_expected": int(len(control)),
                "models": {},
            }
            for model_key in multimodel.MODELS:
                csv_path = REPO / multimodel.RESULTS_DIR / model_key / cfg["csv"]
                existing = _count_existing(csv_path, cfg["target_label"], cfg["control_label"])
                if existing["exists"]:
                    base_df = pd.read_csv(csv_path)
                    base_target_keys = _key_set(base_df[base_df["marker_type"] == cfg["target_label"]], key_cols)
                    base_control_keys = _key_set(base_df[base_df["marker_type"] == cfg["control_label"]], key_cols)
                    supplement_target_keys: set[tuple[Any, ...]] = set()
                    supplement_control_keys: set[tuple[Any, ...]] = set()
                    supplement_csv = getattr(multimodel, "SUPPLEMENT_CSV_BY_SECTION", {}).get(section_key)
                    if supplement_csv:
                        supplement_path = REPO / multimodel.RESULTS_DIR / model_key / supplement_csv
                        if supplement_path.exists():
                            supplement_df = pd.read_csv(supplement_path)
                            supplement_target_keys = _key_set(
                                supplement_df[supplement_df["marker_type"] == cfg["target_label"]], key_cols
                            )
                            supplement_control_keys = _key_set(
                                supplement_df[supplement_df["marker_type"] == cfg["control_label"]], key_cols
                            )
                            existing["supplement_rows"] = int(len(supplement_df))
                    existing["target_remaining_base_only"] = int(len(target_keys - base_target_keys))
                    existing["control_remaining_base_only"] = int(len(control_keys - base_control_keys))
                    existing["target_remaining_with_resume_guard"] = int(
                        len(target_keys - (base_target_keys | supplement_target_keys))
                    )
                    existing["control_remaining_with_resume_guard"] = int(
                        len(control_keys - (base_control_keys | supplement_control_keys))
                    )
                section_counts["models"][model_key] = existing
            base_counts[section_key] = section_counts
        mock_multimodel = await _check_multimodel_mock(multimodel, subsets)

    prompt_parsers = _check_prompt_parsers(multimodel, supplement)
    supplement_mock = await _check_supplement_mock(supplement)

    raw_sources = [_file_state(REPO / path) for path in RAW_SOURCE_CANDIDATES]
    normalized_inputs = [
        _file_state(REPO / "data/sentence_normalized.csv"),
        _file_state(REPO / "data/phrase_normalized.csv"),
        _file_state(REPO / "data/supplement_section1_control_12.csv"),
        _file_state(REPO / "data/supplement_section2_control_465.csv"),
        _file_state(REPO / "data/supplement_section3_control_30.csv"),
    ]

    checks_ok = bool(sentence_exists and sentence_rows > 0)
    checks_ok = checks_ok and all(v["parser_ok"] for group in prompt_parsers.values() for v in group.values())
    checks_ok = checks_ok and all(v.get("has_judgment", False) for v in mock_multimodel.values())
    checks_ok = checks_ok and all(v.get("has_judgment", False) for v in supplement_mock.values() if v.get("input_exists"))

    raw_ready = all(item["exists"] for item in raw_sources if not item["path"].startswith("primary_data"))
    runtime_ready = len(missing_before_stubs) == 0

    status = "pass" if checks_ok else "fail"
    if strict and (not raw_ready or not runtime_ready):
        status = "fail"

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "strict": strict,
        "python": sys.executable,
        "actual_runtime_missing_deps": missing_before_stubs,
        "actual_runtime_ready": runtime_ready,
        "raw_regeneration_ready": raw_ready,
        "raw_sources": raw_sources,
        "normalized_inputs": normalized_inputs,
        "sentence_rows": sentence_rows,
        "base_llm_resume_counts": base_counts,
        "prompt_parser_checks": prompt_parsers,
        "mock_multimodel_checks": mock_multimodel,
        "mock_supplement_checks": supplement_mock,
        "log_path": _rel(LOG_PATH),
    }
    _write_log(payload)
    return (0 if status == "pass" else 1), payload


def main() -> int:
    parser = argparse.ArgumentParser(description="offline preflight for LLM judgment pipelines")
    parser.add_argument("--strict", action="store_true", help="fail if raw sources or runtime deps are missing")
    args = parser.parse_args()
    code, payload = asyncio.run(run_preflight(strict=args.strict))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
