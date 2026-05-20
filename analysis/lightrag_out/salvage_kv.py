"""Salvage: filter cross-contaminated KV stores in each workdir to keep
only that category's own data, and empty the LLM response cache so the
re-run regenerates query answers from the clean graph + VDB.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
CATEGORIES = ["II_니라_X", "IV_라_X", "I_니라_O", "III_라_O"]


def belongs_to(s: str, cat: str) -> bool:
    """Heuristic: does this id/path/source belong to category `cat`?"""
    if not isinstance(s, str):
        return False
    return cat in s


def filter_dict_kv(path: Path, cat: str) -> tuple[int, int]:
    """For dict-of-dict KV files, drop entries whose values reference a
    different category. Returns (kept, dropped)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return (0, 0)

    new = {}
    dropped = 0
    for key, val in data.items():
        # First test by key prefix (e.g. doc IDs are 'CAT__cluster_X')
        if any(c in key for c in CATEGORIES if c != cat):
            dropped += 1
            continue
        # Test by value file_path / source_id / source / file_paths
        ref_strings: list[str] = []
        if isinstance(val, dict):
            for k in ("file_path", "source_id", "source", "full_doc_id", "file_paths"):
                v = val.get(k)
                if isinstance(v, str):
                    ref_strings.append(v)
                elif isinstance(v, list):
                    ref_strings.extend(str(x) for x in v)
        elif isinstance(val, list):
            ref_strings.extend(str(x) for x in val)
        if any(any(c in rs for c in CATEGORIES if c != cat) for rs in ref_strings):
            dropped += 1
            continue
        new[key] = val
    if dropped == 0:
        return (len(new), 0)
    # Write back
    with open(path, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=2)
    return (len(new), dropped)


def empty_cache(path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({}, f)


def salvage_workdir(cat: str) -> dict:
    workdir = BASE / cat
    if not workdir.exists():
        return {"missing": True}
    report = {}
    # KV files to filter. Skip vdb_*.json (already clean) and graph file.
    kv_files = [
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "kv_store_doc_status.json",
        "kv_store_full_entities.json",
        "kv_store_full_relations.json",
        "kv_store_entity_chunks.json",
        "kv_store_relation_chunks.json",
    ]
    for fn in kv_files:
        p = workdir / fn
        if not p.exists():
            continue
        kept, dropped = filter_dict_kv(p, cat)
        report[fn] = f"kept={kept}, dropped={dropped}"

    # Empty the response cache
    cache_path = workdir / "kv_store_llm_response_cache.json"
    if cache_path.exists():
        empty_cache(cache_path)
        report["kv_store_llm_response_cache.json"] = "emptied"

    return report


def main():
    for cat in CATEGORIES:
        print(f"\n=== {cat} ===")
        r = salvage_workdir(cat)
        for k, v in r.items():
            print(f"  {k:45s} {v}")


if __name__ == "__main__":
    main()
