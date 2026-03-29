#!/usr/bin/env python3
"""
Dataset Selection Gate for tool-calling fine-tuning.

This script validates that a dataset truly supports tool calling by computing:
- number of unique tool names
- tool frequency distribution
- % examples with >= K tools available
- % examples with multiple tool calls in ground truth
- % parseable tool catalogs and ground-truth calls

Default target dataset: Salesforce/xlam-function-calling-60k
Dataset fields (per dataset card):
- query: string
- tools: JSON string that decodes to a list[tool]
- answers: JSON string that decodes to a list[{name, arguments}]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import orjson
from tqdm import tqdm

from datasets import load_dataset


@dataclass
class GateThresholds:
    min_examples: int = 1000
    min_unique_tools: int = 200
    min_pct_examples_with_k_tools: float = 0.80   # fraction in [0,1]
    k_tools: int = 5
    min_pct_parseable_tools: float = 0.98
    min_pct_parseable_answers: float = 0.98
    min_pct_nonempty_answers: float = 0.98


@dataclass
class DatasetGateReport:
    dataset_name: str
    split: str
    n_examples_scanned: int

    unique_tool_names: int
    top_tools: List[Tuple[str, int]]

    pct_examples_with_k_tools: float
    pct_examples_with_multi_calls: float

    pct_parseable_tools: float
    pct_parseable_answers: float
    pct_nonempty_answers: float

    tool_count_stats: Dict[str, float]
    answer_call_count_stats: Dict[str, float]

    failures: Dict[str, int]
    passed: bool
    thresholds: Dict[str, Any]


def _safe_json_loads(x: Any) -> Tuple[Optional[Any], bool]:
    """
    Returns: (obj, ok)
    Handles:
      - already-parsed objects
      - JSON strings
    """
    if x is None:
        return None, False
    if isinstance(x, (dict, list)):
        return x, True
    if isinstance(x, (bytes, bytearray)):
        try:
            return orjson.loads(x), True
        except Exception:
            return None, False
    if isinstance(x, str):
        try:
            return json.loads(x), True
        except Exception:
            # sometimes it's JSON-like but not strict; we still mark as fail here
            return None, False
    return None, False


def _basic_stats(arr: List[int]) -> Dict[str, float]:
    if not arr:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    a = np.array(arr, dtype=np.int32)
    return {
        "min": float(np.min(a)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
    }


def run_gate(
    dataset_name: str,
    split: str,
    max_examples: int,
    thresholds: GateThresholds,
    out_dir: Path,
) -> DatasetGateReport:
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name, split=split)

    n_scan = min(len(ds), max_examples) if max_examples > 0 else len(ds)

    tool_name_counter: Counter[str] = Counter()
    failures = defaultdict(int)

    tool_counts = []
    call_counts = []

    parseable_tools = 0
    parseable_answers = 0
    nonempty_answers = 0
    examples_with_k_tools = 0
    examples_with_multi_calls = 0

    for i in tqdm(range(n_scan), desc=f"Scanning {dataset_name}:{split}", ncols=100):
        row = ds[i]
        tools_raw = row.get("tools")
        answers_raw = row.get("answers")

        tools_obj, ok_tools = _safe_json_loads(tools_raw)
        if ok_tools and isinstance(tools_obj, list):
            parseable_tools += 1
            tool_counts.append(len(tools_obj))
            if len(tools_obj) >= thresholds.k_tools:
                examples_with_k_tools += 1
            for t in tools_obj:
                if isinstance(t, dict) and "name" in t and isinstance(t["name"], str):
                    tool_name_counter[t["name"]] += 1
                else:
                    failures["tool_missing_name_or_bad_type"] += 1
        else:
            failures["tools_unparseable_or_not_list"] += 1

        answers_obj, ok_ans = _safe_json_loads(answers_raw)
        if ok_ans and isinstance(answers_obj, list):
            parseable_answers += 1
            call_counts.append(len(answers_obj))
            if len(answers_obj) > 0:
                nonempty_answers += 1
            if len(answers_obj) > 1:
                examples_with_multi_calls += 1
            # sanity-check structure
            for a in answers_obj:
                if not (isinstance(a, dict) and "name" in a and "arguments" in a):
                    failures["answer_bad_shape_missing_keys"] += 1
                else:
                    if not isinstance(a["name"], str):
                        failures["answer_name_not_str"] += 1
                    if not isinstance(a["arguments"], dict):
                        failures["answer_arguments_not_dict"] += 1
        else:
            failures["answers_unparseable_or_not_list"] += 1

    pct_examples_with_k_tools = examples_with_k_tools / max(n_scan, 1)
    pct_examples_with_multi_calls = examples_with_multi_calls / max(n_scan, 1)
    pct_parseable_tools = parseable_tools / max(n_scan, 1)
    pct_parseable_answers = parseable_answers / max(n_scan, 1)
    pct_nonempty_answers = nonempty_answers / max(n_scan, 1)

    unique_tool_names = len(tool_name_counter)
    top_tools = tool_name_counter.most_common(20)

    # Decide pass/fail
    checks = {
        "min_examples": (n_scan >= thresholds.min_examples),
        "min_unique_tools": (unique_tool_names >= thresholds.min_unique_tools),
        "min_pct_examples_with_k_tools": (pct_examples_with_k_tools >= thresholds.min_pct_examples_with_k_tools),
        "min_pct_parseable_tools": (pct_parseable_tools >= thresholds.min_pct_parseable_tools),
        "min_pct_parseable_answers": (pct_parseable_answers >= thresholds.min_pct_parseable_answers),
        "min_pct_nonempty_answers": (pct_nonempty_answers >= thresholds.min_pct_nonempty_answers),
    }
    passed = all(checks.values())

    report = DatasetGateReport(
        dataset_name=dataset_name,
        split=split,
        n_examples_scanned=n_scan,
        unique_tool_names=unique_tool_names,
        top_tools=top_tools,
        pct_examples_with_k_tools=pct_examples_with_k_tools,
        pct_examples_with_multi_calls=pct_examples_with_multi_calls,
        pct_parseable_tools=pct_parseable_tools,
        pct_parseable_answers=pct_parseable_answers,
        pct_nonempty_answers=pct_nonempty_answers,
        tool_count_stats=_basic_stats(tool_counts),
        answer_call_count_stats=_basic_stats(call_counts),
        failures=dict(sorted(failures.items(), key=lambda kv: -kv[1])),
        passed=passed,
        thresholds=asdict(thresholds),
    )

    # Write artifacts
    (out_dir / "dataset_report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    # Also dump tool histogram
    if unique_tool_names > 0:
        df = pd.DataFrame(tool_name_counter.most_common(), columns=["tool_name", "count"])
        df.to_csv(out_dir / "tool_name_counts.csv", index=False)

    # Pretty console summary
    print("\n=== DATASET GATE SUMMARY ===")
    print(f"Dataset: {dataset_name} | Split: {split} | Scanned: {n_scan}")
    print(f"Unique tool names: {unique_tool_names}")
    print(f"% examples with >= {thresholds.k_tools} tools: {pct_examples_with_k_tools:.3f}")
    print(f"% examples with multiple GT calls: {pct_examples_with_multi_calls:.3f}")
    print(f"% parseable tools: {pct_parseable_tools:.3f}")
    print(f"% parseable answers: {pct_parseable_answers:.3f}")
    print(f"% nonempty answers: {pct_nonempty_answers:.3f}")
    print(f"PASSED: {passed}")
    print("Top-10 tools:", top_tools[:10])
    if failures:
        print("Top failures:", list(report.failures.items())[:8])
    print("============================\n")

    # If failed, print which checks failed
    if not passed:
        print("Failed checks:")
        for k, ok in checks.items():
            if not ok:
                print(f" - {k} (threshold={getattr(thresholds, k) if hasattr(thresholds, k) else thresholds.min_pct_examples_with_k_tools})")

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="Salesforce/xlam-function-calling-60k")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_examples", type=int, default=20000, help="0 = scan full split")
    ap.add_argument("--out_dir", type=str, default="artifacts/step1_dataset_gate")

    ap.add_argument("--min_examples", type=int, default=1000)
    ap.add_argument("--min_unique_tools", type=int, default=200)
    ap.add_argument("--k_tools", type=int, default=5)
    ap.add_argument("--min_pct_examples_with_k_tools", type=float, default=0.80)
    ap.add_argument("--min_pct_parseable_tools", type=float, default=0.98)
    ap.add_argument("--min_pct_parseable_answers", type=float, default=0.98)
    ap.add_argument("--min_pct_nonempty_answers", type=float, default=0.98)

    args = ap.parse_args()

    thresholds = GateThresholds(
        min_examples=args.min_examples,
        min_unique_tools=args.min_unique_tools,
        k_tools=args.k_tools,
        min_pct_examples_with_k_tools=args.min_pct_examples_with_k_tools,
        min_pct_parseable_tools=args.min_pct_parseable_tools,
        min_pct_parseable_answers=args.min_pct_parseable_answers,
        min_pct_nonempty_answers=args.min_pct_nonempty_answers,
    )

    run_gate(
        dataset_name=args.dataset,
        split=args.split,
        max_examples=args.max_examples,
        thresholds=thresholds,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()