from __future__ import annotations

import json
from typing import Any, Dict, List, Set, Tuple


def normalize_calls_for_scoring(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only the exact expected fields for scoring.
    """
    out = []
    for c in calls:
        if isinstance(c, dict) and isinstance(c.get("name"), str) and isinstance(c.get("arguments"), dict):
            out.append({"name": c["name"], "arguments": c["arguments"]})
    return out


def _canonicalize(value: Any) -> Any:
    """
    Convert nested lists/dicts into hashable canonical structures.
    """
    if isinstance(value, dict):
        return tuple((k, _canonicalize(v)) for k, v in sorted(value.items(), key=lambda kv: kv[0]))
    if isinstance(value, list):
        return tuple(_canonicalize(x) for x in value)
    return value


def sorted_call_signature(call: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    """
    Exact call signature for strict matching.
    Handles nested lists/dicts in arguments by converting them
    into hashable canonical tuples.
    """
    name = call["name"]
    args = call["arguments"]
    return name, tuple((k, _canonicalize(v)) for k, v in sorted(args.items(), key=lambda kv: kv[0]))


def tool_name_set(calls: List[Dict[str, Any]]) -> Set[str]:
    return {c["name"] for c in calls}


def arg_key_pairs(calls: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    """
    Returns set of (tool_name, arg_key)
    """
    out: Set[Tuple[str, str]] = set()
    for c in calls:
        name = c["name"]
        for k in c["arguments"].keys():
            out.add((name, k))
    return out


def exact_call_set(calls: List[Dict[str, Any]]) -> Set[Tuple[str, Tuple[Tuple[str, Any], ...]]]:
    return {sorted_call_signature(c) for c in calls}


def safe_div(x: float, y: float) -> float:
    return x / y if y != 0 else 0.0


def prf1(tp: int, pred_n: int, gold_n: int) -> Dict[str, float]:
    p = safe_div(tp, pred_n)
    r = safe_div(tp, gold_n)
    f1 = safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}