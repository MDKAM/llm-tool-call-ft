from __future__ import annotations

import json
from typing import Any, Dict, List

from src.prompting import build_system_message
from src.validator import validate_output


def load_tools(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_demo_messages(user_prompt: str, tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system_text = build_system_message(tools)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_prompt},
    ]


def pretty_print_tools(tools: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, tool in enumerate(tools, start=1):
        lines.append(f"{i}. {tool['name']}")
        lines.append(f"   Description: {tool.get('description', '')}")
        params = tool.get("parameters", {})
        if params:
            lines.append("   Parameters:")
            for p_name, spec in params.items():
                req = "required" if spec.get("required", False) else "optional"
                p_type = spec.get("type", "any")
                p_desc = spec.get("description", "")
                lines.append(f"     - {p_name} ({p_type}, {req}): {p_desc}")
        else:
            lines.append("   Parameters: none")
    return "\n".join(lines)


def validate_prediction_text(prediction_text: str, tools: List[Dict[str, Any]], lenient_types: bool = True):
    return validate_output(prediction_text, tools, lenient_types=lenient_types)


def mock_execute_calls(calls: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for i, call in enumerate(calls, start=1):
        lines.append(f"[MOCK] Call {i}: {call['name']} with arguments {json.dumps(call['arguments'], ensure_ascii=False)}")
    return lines