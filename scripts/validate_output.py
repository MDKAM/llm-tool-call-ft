#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datasets import load_dataset

from src.validator import validate_output


def safe_load_json(x):
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        return json.loads(x)
    raise TypeError(f"Unsupported JSON type: {type(x)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="Salesforce/xlam-function-calling-60k")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--output_text", type=str, required=True)
    ap.add_argument("--allow_extra_arguments", action="store_true")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    row = ds[args.index]
    tools = safe_load_json(row["tools"])

    result, parsed = validate_output(
        args.output_text,
        tools,
        allow_extra_arguments=args.allow_extra_arguments,
    )

    print(json.dumps(result.__dict__, indent=2))
    if parsed is not None:
        print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()