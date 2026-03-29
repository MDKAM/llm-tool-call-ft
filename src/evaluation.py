from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .metrics import (
    normalize_calls_for_scoring,
    tool_name_set,
    arg_key_pairs,
    exact_call_set,
    prf1,
)
from .validator import validate_output


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_gold_from_record(record: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """
    Returns:
      gold_calls, tools, user_query
    """
    msgs = record["messages"]
    system_text = msgs[0]["content"]
    user_text = msgs[1]["content"]
    assistant_text = msgs[2]["content"]

    # recover tools from system prompt
    marker = "TOOL CATALOG (JSON):\n"
    if marker not in system_text:
        raise ValueError("Could not find tool catalog in system message.")
    tools_json = system_text.split(marker, 1)[1]
    tools = json.loads(tools_json)

    gold_calls = json.loads(assistant_text)
    return gold_calls, tools, user_text


def evaluate_records(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
    *,
    allow_extra_arguments: bool = False,
    lenient_types: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    pred_records format:
      [{"prediction": "<raw model text>"}]
    """
    assert len(gold_records) == len(pred_records), "gold and prediction lengths must match"

    n = len(gold_records)

    json_parse_ok = 0
    array_ok = 0
    call_shape_ok = 0
    strict_valid_ok = 0

    tool_name_exact = 0
    arg_exact = 0
    joint_exact = 0

    total_tool_tp = 0
    total_tool_pred = 0
    total_tool_gold = 0

    total_argkey_tp = 0
    total_argkey_pred = 0
    total_argkey_gold = 0

    per_example_rows: List[Dict[str, Any]] = []

    for i, (gold_rec, pred_rec) in enumerate(zip(gold_records, pred_records)):
        gold_calls, tools, user_text = extract_gold_from_record(gold_rec)
        gold_calls = normalize_calls_for_scoring(gold_calls)

        pred_text = pred_rec["prediction"]

        result, parsed = validate_output(
            pred_text,
            tools,
            allow_extra_arguments=allow_extra_arguments,
            lenient_types=lenient_types,
        )

        row = {
            "index": i,
            "user": user_text,
            "prediction": pred_text,
            "json_parse_ok": result.json_parse_ok,
            "array_ok": result.array_ok,
            "call_shape_ok": result.call_shape_ok,
            "strict_valid_ok": result.ok,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }

        if result.json_parse_ok:
            json_parse_ok += 1
        if result.array_ok:
            array_ok += 1
        if result.call_shape_ok:
            call_shape_ok += 1
        if result.ok:
            strict_valid_ok += 1

        pred_calls = parsed if parsed is not None else []
        pred_calls = normalize_calls_for_scoring(pred_calls)

        gold_tool_set = tool_name_set(gold_calls)
        pred_tool_set = tool_name_set(pred_calls)

        gold_argkey_set = arg_key_pairs(gold_calls)
        pred_argkey_set = arg_key_pairs(pred_calls)

        gold_exact_set = exact_call_set(gold_calls)
        pred_exact_set = exact_call_set(pred_calls)

        tool_tp = len(gold_tool_set & pred_tool_set)
        argkey_tp = len(gold_argkey_set & pred_argkey_set)

        total_tool_tp += tool_tp
        total_tool_pred += len(pred_tool_set)
        total_tool_gold += len(gold_tool_set)

        total_argkey_tp += argkey_tp
        total_argkey_pred += len(pred_argkey_set)
        total_argkey_gold += len(gold_argkey_set)

        tool_exact_this = int(pred_tool_set == gold_tool_set)
        arg_exact_this = int(pred_exact_set == gold_exact_set)
        joint_exact_this = int((pred_tool_set == gold_tool_set) and (pred_exact_set == gold_exact_set))

        tool_name_exact += tool_exact_this
        arg_exact += arg_exact_this
        joint_exact += joint_exact_this

        row["gold_calls"] = gold_calls
        row["parsed_prediction_calls"] = pred_calls
        row["tool_exact"] = bool(tool_exact_this)
        row["arg_exact"] = bool(arg_exact_this)
        row["joint_exact"] = bool(joint_exact_this)

        per_example_rows.append(row)

    tool_prf1 = prf1(total_tool_tp, total_tool_pred, total_tool_gold)
    argkey_prf1 = prf1(total_argkey_tp, total_argkey_pred, total_argkey_gold)

    metrics = {
        "n_examples": n,
        "json_parse_rate": json_parse_ok / n if n else 0.0,
        "array_rate": array_ok / n if n else 0.0,
        "call_shape_rate": call_shape_ok / n if n else 0.0,
        "strict_valid_rate": strict_valid_ok / n if n else 0.0,
        "tool_name_exact_rate": tool_name_exact / n if n else 0.0,
        "argument_exact_rate": arg_exact / n if n else 0.0,
        "joint_exact_rate": joint_exact / n if n else 0.0,
        "tool_name_precision": tool_prf1["precision"],
        "tool_name_recall": tool_prf1["recall"],
        "tool_name_f1": tool_prf1["f1"],
        "arg_key_precision": argkey_prf1["precision"],
        "arg_key_recall": argkey_prf1["recall"],
        "arg_key_f1": argkey_prf1["f1"],
    }

    return metrics, per_example_rows