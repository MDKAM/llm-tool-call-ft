#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import read_jsonl, evaluate_records
from src.analysis import compute_per_tool_metrics, summarize_errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_path", type=str, default="data/test.jsonl")
    ap.add_argument("--pred_path", type=str, required=True)
    ap.add_argument("--metrics_out", type=str, default="artifacts/final_metrics.json")
    ap.add_argument("--per_tool_out", type=str, default="artifacts/per_tool_metrics.json")
    ap.add_argument("--error_summary_out", type=str, default="artifacts/error_analysis.json")
    ap.add_argument("--error_examples_out", type=str, default="artifacts/error_examples.json")
    ap.add_argument("--allow_extra_arguments", action="store_true")
    ap.add_argument("--lenient_types", action="store_true")
    ap.add_argument("--max_examples", type=int, default=0)
    args = ap.parse_args()

    gold_records = read_jsonl(args.gold_path)
    pred_records = read_jsonl(args.pred_path)

    if args.max_examples and args.max_examples > 0:
        gold_records = gold_records[:args.max_examples]
        pred_records = pred_records[:args.max_examples]

    if len(gold_records) != len(pred_records):
        raise ValueError(f"gold/pred length mismatch: gold={len(gold_records)} pred={len(pred_records)}")

    metrics, rows = evaluate_records(
        gold_records,
        pred_records,
        allow_extra_arguments=args.allow_extra_arguments,
        lenient_types=args.lenient_types,
    )

    per_tool = compute_per_tool_metrics(rows)
    error_counts, error_examples = summarize_errors(rows)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(args.per_tool_out, "w", encoding="utf-8") as f:
        json.dump(per_tool, f, indent=2)

    with open(args.error_summary_out, "w", encoding="utf-8") as f:
        json.dump(error_counts, f, indent=2)

    with open(args.error_examples_out, "w", encoding="utf-8") as f:
        json.dump(error_examples, f, indent=2)

    print("=== METRICS ===")
    print(json.dumps(metrics, indent=2))
    print("\n=== ERROR COUNTS ===")
    print(json.dumps(error_counts, indent=2))
    print(f"\nSaved per-tool metrics to: {args.per_tool_out}")


if __name__ == "__main__":
    main()