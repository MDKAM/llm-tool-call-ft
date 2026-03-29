#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import read_jsonl, evaluate_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_path", type=str, default="data/test.jsonl")
    ap.add_argument("--pred_path", type=str, required=True)
    ap.add_argument("--metrics_out", type=str, default="artifacts/baseline_metrics.json")
    ap.add_argument("--rows_out", type=str, default="artifacts/baseline_predictions.jsonl")
    ap.add_argument("--allow_extra_arguments", action="store_true")
    ap.add_argument("--lenient_types", action="store_true")
    ap.add_argument("--max_examples", type=int, default=0, help="0 means use all rows")
    args = ap.parse_args()

    gold_records = read_jsonl(args.gold_path)
    pred_records = read_jsonl(args.pred_path)

    if args.max_examples and args.max_examples > 0:
        gold_records = gold_records[:args.max_examples]
        pred_records = pred_records[:args.max_examples]

    if len(gold_records) != len(pred_records):
        raise ValueError(
            f"gold/pred length mismatch: gold={len(gold_records)} pred={len(pred_records)}. "
            f"Use matching files or pass --max_examples."
        )

    metrics, rows = evaluate_records(
        gold_records,
        pred_records,
        allow_extra_arguments=args.allow_extra_arguments,
        lenient_types=args.lenient_types,
    )

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rows_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(args.rows_out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()