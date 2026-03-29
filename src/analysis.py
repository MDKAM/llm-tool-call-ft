from __future__ import annotations

from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

from .metrics import (
    normalize_calls_for_scoring,
    tool_name_set,
    arg_key_pairs,
    exact_call_set,
)


def categorize_error(row: Dict[str, Any]) -> str:
    """
    Assign a single primary error category to each prediction row.
    """
    if not row.get("json_parse_ok", False):
        return "json_parse_error"
    if not row.get("array_ok", False):
        return "not_array"
    if not row.get("call_shape_ok", False):
        return "bad_call_shape"
    if not row.get("strict_valid_ok", False):
        err = row.get("error_type")
        if err == "unknown_tool":
            return "unknown_tool"
        if err == "bad_arguments":
            return "bad_arguments"
        return "validation_error"

    if row.get("joint_exact", False):
        return "correct"

    if not row.get("tool_exact", False):
        return "wrong_tool_selection"

    if not row.get("arg_exact", False):
        gold_calls = normalize_calls_for_scoring(row.get("gold_calls", []))
        pred_calls = normalize_calls_for_scoring(row.get("parsed_prediction_calls", []))

        gold_argkeys = arg_key_pairs(gold_calls)
        pred_argkeys = arg_key_pairs(pred_calls)

        if gold_argkeys != pred_argkeys:
            return "wrong_argument_keys"

        gold_exact = exact_call_set(gold_calls)
        pred_exact = exact_call_set(pred_calls)
        if gold_exact != pred_exact:
            return "wrong_argument_values"

    return "other"


def compute_per_tool_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Per-tool support and exact-match style stats.
    Tool counted if present in gold.
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "support": 0,
        "predicted": 0,
        "tool_tp": 0,
        "arg_exact_tp": 0,
        "joint_exact_tp": 0,
    })

    for row in rows:
        gold_calls = normalize_calls_for_scoring(row.get("gold_calls", []))
        pred_calls = normalize_calls_for_scoring(row.get("parsed_prediction_calls", []))

        gold_by_tool = {c["name"]: c for c in gold_calls}
        pred_by_tool = {c["name"]: c for c in pred_calls}

        all_tools = set(gold_by_tool.keys()) | set(pred_by_tool.keys())

        for tool in all_tools:
            if tool in gold_by_tool:
                stats[tool]["support"] += 1
            if tool in pred_by_tool:
                stats[tool]["predicted"] += 1
            if tool in gold_by_tool and tool in pred_by_tool:
                stats[tool]["tool_tp"] += 1
                if gold_by_tool[tool] == pred_by_tool[tool]:
                    stats[tool]["arg_exact_tp"] += 1
                    stats[tool]["joint_exact_tp"] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for tool, s in stats.items():
        support = s["support"]
        predicted = s["predicted"]
        tp = s["tool_tp"]
        arg_exact_tp = s["arg_exact_tp"]
        joint_exact_tp = s["joint_exact_tp"]

        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        out[tool] = {
            "support": support,
            "predicted": predicted,
            "tool_precision": precision,
            "tool_recall": recall,
            "tool_f1": f1,
            "arg_exact_rate_given_gold_tool": (arg_exact_tp / support) if support else 0.0,
            "joint_exact_rate_given_gold_tool": (joint_exact_tp / support) if support else 0.0,
        }

    return dict(sorted(out.items(), key=lambda kv: (-kv[1]["support"], kv[0])))


def summarize_errors(rows: List[Dict[str, Any]], max_examples_per_type: int = 5) -> Tuple[Dict[str, int], Dict[str, List[Dict[str, Any]]]]:
    counts = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        category = categorize_error(row)
        counts[category] += 1

        if len(examples[category]) < max_examples_per_type:
            examples[category].append({
                "index": row.get("index"),
                "user": row.get("user"),
                "error_type": row.get("error_type"),
                "error_message": row.get("error_message"),
                "gold_calls": row.get("gold_calls"),
                "parsed_prediction_calls": row.get("parsed_prediction_calls"),
                "prediction": row.get("prediction"),
            })

    return dict(counts), dict(examples)