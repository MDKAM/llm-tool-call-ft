#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from datasets import load_dataset

from src.prompting import build_system_message
from src.preprocess import safe_json_loads, answers_to_model_output, dumps_json_array_only
from src.validator import validate_output


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_record(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tools, ok_tools = safe_json_loads(row.get("tools"))
    answers, ok_answers = safe_json_loads(row.get("answers"))
    query = row.get("query")

    if not (ok_tools and isinstance(tools, list)):
        raise ValueError("tools_unparseable_or_not_list")
    if not (ok_answers and isinstance(answers, list)):
        raise ValueError("answers_unparseable_or_not_list")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query_missing_or_empty")

    model_calls = answers_to_model_output(answers)
    assistant_text = dumps_json_array_only(model_calls)

    # Validate GT against the given tool catalog using Step-2 validator
    result, _ = validate_output(assistant_text, tools, lenient_types=True)
    if not result.ok:
        raise ValueError(f"gt_validation_failed:{result.error_type}:{result.error_message}")

    system_text = build_system_message(tools)

    record = {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_text},
        ]
    }

    meta = {"n_tools": len(tools), "n_calls": len(model_calls)}
    return record, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="Salesforce/xlam-function-calling-60k")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_examples", type=int, default=60000, help="0 = all")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--test_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_data_dir", type=str, default="data")
    ap.add_argument("--out_artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--sample_n", type=int, default=5)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    n_total = len(ds)
    n_use = n_total if args.max_examples == 0 else min(n_total, args.max_examples)

    train_idx, val_idx, test_idx = split_indices(n_use, args.val_ratio, args.test_ratio, args.seed)

    data_dir = Path(args.out_data_dir)
    art_dir = Path(args.out_artifacts_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    stats = {"dataset": args.dataset, "split": args.split, "n_total": n_total, "n_used": n_use, "splits": {}}
    samples: List[Dict[str, Any]] = []

    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        records: List[Dict[str, Any]] = []
        kept = dropped = 0
        tool_counts: List[int] = []
        call_counts: List[int] = []
        drop_reasons: Dict[str, int] = {}

        for i in tqdm(indices, desc=f"Building {split_name}", ncols=100):
            try:
                rec, meta = build_record(ds[i])
                records.append(rec)
                kept += 1
                tool_counts.append(meta["n_tools"])
                call_counts.append(meta["n_calls"])

                if split_name == "train" and len(samples) < args.sample_n:
                    samples.append(
                        {
                            "user": rec["messages"][1]["content"],
                            "assistant": rec["messages"][2]["content"],
                            "system_preview": rec["messages"][0]["content"][:600] + " ...",
                            "n_tools": meta["n_tools"],
                            "n_calls": meta["n_calls"],
                        }
                    )
            except Exception as e:
                dropped += 1
                key = str(e)[:160]
                drop_reasons[key] = drop_reasons.get(key, 0) + 1

        write_jsonl(data_dir / f"{split_name}.jsonl", records)

        stats["splits"][split_name] = {
            "kept": kept,
            "dropped": dropped,
            "avg_tools": (sum(tool_counts) / len(tool_counts)) if tool_counts else 0.0,
            "avg_calls": (sum(call_counts) / len(call_counts)) if call_counts else 0.0,
            "tool_min": min(tool_counts) if tool_counts else 0,
            "tool_max": max(tool_counts) if tool_counts else 0,
            "calls_min": min(call_counts) if call_counts else 0,
            "calls_max": max(call_counts) if call_counts else 0,
            "top_drop_reasons": sorted(drop_reasons.items(), key=lambda kv: -kv[1])[:10],
        }

    (art_dir / "data_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (art_dir / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()