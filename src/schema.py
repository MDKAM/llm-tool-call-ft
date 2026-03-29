from __future__ import annotations

import ast
import re
from typing import Any, Dict, Tuple

_ALIASES = {
    "string": "str",
    "boolean": "bool",
    "integer": "int",
    "number": "float",
    "array": "list",
    "object": "dict",
}

_GENERIC_RE = re.compile(r"^\s*([A-Za-z_]+)\s*\[(.*)\]\s*$")


def is_call_obj(x: Any) -> bool:
    return (
        isinstance(x, dict)
        and set(x.keys()) == {"name", "arguments"}
        and isinstance(x.get("name"), str)
        and isinstance(x.get("arguments"), dict)
    )


def _canon_token(tok: str) -> str:
    tok = tok.strip()
    if not tok:
        return "any"
    t = tok.lower()
    return _ALIASES.get(t, t)


def normalize_type(t: Any) -> str:
    if t is None:
        return "any"
    s = str(t).strip()
    if not s:
        return "any"

    m = _GENERIC_RE.match(s)
    if not m:
        return _canon_token(s)

    head = _canon_token(m.group(1))
    inner = m.group(2).strip()
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    parts = [_canon_token(p) for p in parts]
    return f"{head}[{','.join(parts)}]"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_float(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _can_parse_float_string(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    try:
        float(v.strip())
        return True
    except Exception:
        return False


def _try_parse_literal(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return v
    try:
        return ast.literal_eval(s)
    except Exception:
        return v


def _type_ok_simple(expected: str, value: Any, *, lenient: bool = False) -> bool:
    if expected == "int":
        if _is_int(value):
            return True
        if lenient and isinstance(value, float):
            return True
        if lenient and _can_parse_float_string(value):
            return True
        return False

    if expected == "float":
        if _is_float(value):
            return True
        if lenient and _can_parse_float_string(value):
            return True
        return False

    if expected == "str":
        if isinstance(value, str):
            return True
        if lenient and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        return False

    if expected == "bool":
        return isinstance(value, bool)

    if expected == "list":
        if isinstance(value, list):
            return True
        if lenient:
            parsed = _try_parse_literal(value)
            return isinstance(parsed, list)
        return False

    if expected == "dict":
        if isinstance(value, dict):
            return True
        if lenient:
            parsed = _try_parse_literal(value)
            return isinstance(parsed, dict)
        return False

    return True


def type_ok(expected_raw: Any, value: Any, *, lenient: bool = False) -> bool:
    expected = normalize_type(expected_raw)

    if expected.startswith("optional[") and expected.endswith("]"):
        if value is None:
            return True
        inner = expected[len("optional["):-1]
        return type_ok(inner, value, lenient=lenient)

    if expected.startswith("union[") and expected.endswith("]"):
        inner = expected[len("union["):-1]
        branches = [b.strip() for b in inner.split(",") if b.strip()]
        return any(type_ok(b, value, lenient=lenient) for b in branches)

    if expected.startswith("list[") and expected.endswith("]"):
        inner = expected[len("list["):-1].strip()
        parsed = value
        if lenient and isinstance(value, str):
            parsed = _try_parse_literal(value)
        if not isinstance(parsed, list):
            return False

        inner_norm = normalize_type(inner)
        if inner_norm in {"int", "float", "str", "bool"} and len(parsed) > 0:
            return all(_type_ok_simple(inner_norm, x, lenient=lenient) for x in parsed)
        return True

    if expected.startswith("dict[") and expected.endswith("]"):
        parsed = value
        if lenient and isinstance(value, str):
            parsed = _try_parse_literal(value)
        if not isinstance(parsed, dict):
            return False
        return True

    return _type_ok_simple(expected, value, lenient=lenient)


def normalize_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(tool, dict):
        raise TypeError("tool must be dict")

    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool missing valid name")

    desc = tool.get("description", "")
    if desc is None:
        desc = ""
    if not isinstance(desc, str):
        desc = str(desc)

    params = tool.get("parameters", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"tool '{name}' parameters must be dict")

    norm_params: Dict[str, Dict[str, Any]] = {}
    for p_name, spec in params.items():
        if not isinstance(p_name, str) or not p_name:
            raise ValueError(f"tool '{name}' invalid param name: {p_name}")
        if not isinstance(spec, dict):
            raise ValueError(f"tool '{name}' param '{p_name}' spec must be dict")

        p_type = normalize_type(spec.get("type"))
        p_desc = spec.get("description", "")
        if p_desc is None:
            p_desc = ""
        if not isinstance(p_desc, str):
            p_desc = str(p_desc)

        p_req = bool(spec.get("required", False))
        norm_params[p_name] = {"type": p_type, "description": p_desc, "required": p_req}

    return {"name": name, "description": desc, "parameters": norm_params}


def validate_arguments(
    tool_param_spec: Dict[str, Dict[str, Any]],
    arguments: Dict[str, Any],
    *,
    allow_extra: bool = False,
    lenient_types: bool = False,
) -> Tuple[bool, str]:
    required = {k for k, spec in tool_param_spec.items() if spec.get("required", False)}
    missing = sorted([k for k in required if k not in arguments])
    if missing:
        return False, f"missing_required={missing}"

    if not allow_extra:
        extra = sorted([k for k in arguments.keys() if k not in tool_param_spec])
        if extra:
            return False, f"extra_not_allowed={extra}"

    for k, v in arguments.items():
        spec = tool_param_spec.get(k)
        if spec is None:
            continue
        exp_t = spec.get("type", "any")
        if not type_ok(exp_t, v, lenient=lenient_types):
            return False, f"type_mismatch key='{k}' expected='{exp_t}' got='{type(v).__name__}'"

    return True, "ok"