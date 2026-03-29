from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .schema import is_call_obj, normalize_tool, validate_arguments


@dataclass
class ValidationResult:
    ok: bool
    json_parse_ok: bool
    array_ok: bool
    call_shape_ok: bool
    tool_name_ok: bool
    arguments_ok: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def build_tool_index(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for t in tools:
        nt = normalize_tool(t)
        idx[nt["name"]] = nt
    return idx


def validate_output(
    output_text: str,
    tools: List[Dict[str, Any]],
    *,
    allow_extra_arguments: bool = False,
    lenient_types: bool = False,
) -> Tuple[ValidationResult, Optional[List[Dict[str, Any]]]]:
    # JSON parse
    try:
        parsed = json.loads(output_text)
    except Exception as e:
        return (
            ValidationResult(
                ok=False,
                json_parse_ok=False,
                array_ok=False,
                call_shape_ok=False,
                tool_name_ok=False,
                arguments_ok=False,
                error_type="json_parse_error",
                error_message=str(e),
            ),
            None,
        )

    # must be list
    if not isinstance(parsed, list):
        return (
            ValidationResult(
                ok=False,
                json_parse_ok=True,
                array_ok=False,
                call_shape_ok=False,
                tool_name_ok=False,
                arguments_ok=False,
                error_type="not_array",
                error_message="Output must be a JSON array (top-level list).",
            ),
            None,
        )

    # validate call shapes
    for i, call in enumerate(parsed):
        if not is_call_obj(call):
            return (
                ValidationResult(
                    ok=False,
                    json_parse_ok=True,
                    array_ok=True,
                    call_shape_ok=False,
                    tool_name_ok=False,
                    arguments_ok=False,
                    error_type="bad_call_shape",
                    error_message=f"Item {i} must be EXACTLY {{'name','arguments'}} with name=str, arguments=dict.",
                ),
                None,
            )

    tool_index = build_tool_index(tools)

    # validate each call
    for i, call in enumerate(parsed):
        name = call["name"]
        args = call["arguments"]

        tool = tool_index.get(name)
        if tool is None:
            return (
                ValidationResult(
                    ok=False,
                    json_parse_ok=True,
                    array_ok=True,
                    call_shape_ok=True,
                    tool_name_ok=False,
                    arguments_ok=False,
                    error_type="unknown_tool",
                    error_message=f"Item {i}: tool '{name}' not found in provided tool catalog.",
                ),
                parsed,
            )

        ok_args, msg = validate_arguments(tool["parameters"], args, allow_extra=allow_extra_arguments, lenient_types=lenient_types)
        if not ok_args:
            return (
                ValidationResult(
                    ok=False,
                    json_parse_ok=True,
                    array_ok=True,
                    call_shape_ok=True,
                    tool_name_ok=True,
                    arguments_ok=False,
                    error_type="bad_arguments",
                    error_message=f"Item {i}: {msg}",
                ),
                parsed,
            )

    return (
        ValidationResult(
            ok=True,
            json_parse_ok=True,
            array_ok=True,
            call_shape_ok=True,
            tool_name_ok=True,
            arguments_ok=True,
        ),
        parsed,
    )