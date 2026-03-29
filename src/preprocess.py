from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional


def safe_json_loads(x: Any) -> Tuple[Optional[Any], bool]:
    if x is None:
        return None, False
    if isinstance(x, (dict, list)):
        return x, True
    if isinstance(x, str):
        try:
            return json.loads(x), True
        except Exception:
            return None, False
    return None, False


def answers_to_model_output(answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dataset answers format:
      [{"name": "...", "arguments": {...}}, ...]
    Model output format (same keys):
      [{"name": "...", "arguments": {...}}, ...]
    We enforce strict shape per call, dropping malformed items.
    """
    out: List[Dict[str, Any]] = []
    for a in answers:
        if not isinstance(a, dict):
            continue
        name = a.get("name")
        args = a.get("arguments")
        if isinstance(name, str) and isinstance(args, dict):
            out.append({"name": name, "arguments": args})
    return out


def dumps_json_array_only(obj: List[Dict[str, Any]]) -> str:
    """
    Must return ONLY a JSON array string, no whitespace constraints needed,
    but we keep it compact.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))