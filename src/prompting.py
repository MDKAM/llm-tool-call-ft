from __future__ import annotations

import json
from typing import Any, Dict, List


STRICT_INSTRUCTIONS = """You are a tool-calling assistant.

You MUST output ONLY a JSON array of tool calls with NO extra text.

Each tool call MUST have EXACTLY this shape:
{"name": "<tool_name>", "arguments": { ... }}

Rules:
- Output must be valid JSON.
- Output must be a JSON array (possibly empty).
- Each array item must contain ONLY keys: "name" and "arguments".
- "name" must be one of the provided tools.
- "arguments" must match the selected tool's parameter requirements:
  * include all required parameters
  * do not include unknown parameters
  * basic type checks should hold (int/float/list/str/bool/dict)
"""


def tool_catalog_to_json(tools: List[Dict[str, Any]]) -> str:
    """
    Preserve dataset tool structure (name, description, parameters spec).
    We embed as JSON for clarity and faithful supervision.
    """
    # Keep exactly the dataset keys we care about.
    compact = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        compact.append(
            {
                "name": t.get("name"),
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            }
        )
    return json.dumps(compact, ensure_ascii=False)


def build_system_message(tools: List[Dict[str, Any]]) -> str:
    return STRICT_INSTRUCTIONS + "\n\nTOOL CATALOG (JSON):\n" + tool_catalog_to_json(tools)