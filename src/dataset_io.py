from __future__ import annotations

import json
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_rows(rows: List[Dict[str, Any]], max_examples: int) -> List[Dict[str, Any]]:
    if max_examples is None or max_examples <= 0:
        return rows
    return rows[:max_examples]


def record_to_messages(record: Dict[str, Any], include_assistant: bool = True) -> List[Dict[str, str]]:
    msgs = record["messages"]
    if include_assistant:
        return msgs
    return msgs[:2]


def assistant_text_from_record(record: Dict[str, Any]) -> str:
    return record["messages"][2]["content"]