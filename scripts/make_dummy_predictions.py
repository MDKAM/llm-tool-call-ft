#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_path", type=str, default="data/test.jsonl")
    ap.add_argument("--out_path", type=str, default="artifacts/dummy_predictions.jsonl")
    ap.add_argument(
        "--mode",
        type=str,
        default="empty",
        choices=["empty", "invalid_json", "echo_gold"],
        help="empty -> []; invalid_json -> malformed; echo_gold -> oracle sanity check",
    )
    args = ap.parse_args()

    gold = read_jsonl(args.gold_path)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        for rec in gold:
            assistant_text = rec["messages"][2]["content"]

            if args.mode == "empty":
                pred = "[]"
            elif args.mode == "invalid_json":
                pred = "[{]"
            elif args.mode == "echo_gold":
                pred = assistant_text
            else:
                raise ValueError(args.mode)

            f.write(json.dumps({"prediction": pred}, ensure_ascii=False) + "\n")

    print(f"Wrote predictions to {args.out_path}")


if __name__ == "__main__":
    main()