from __future__ import annotations

import json
import re
from typing import Any


NONE_LIKE_VALUES = {
    "",
    "none",
    "no match",
    "no_match",
    "null",
    "skipped",
    "n/a",
    "na",
    "not applicable",
}
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return json.dumps(value, ensure_ascii=True, sort_keys=True).strip()


def normalize_text(value: Any) -> str:
    return " ".join(as_text(value).lower().replace("_", " ").split())


def is_none_like(value: Any) -> bool:
    text = normalize_text(value)
    return text in NONE_LIKE_VALUES


def strip_reasoning(value: str) -> str:
    text = THINK_RE.sub("", value).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    return text


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return strip_reasoning(value.strip())
    if isinstance(value, list):
        parts = [extract_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("content", "text", "output", "expected", "value"):
            if key in value:
                text = extract_text(value.get(key))
                if text:
                    return text
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            text = extract_text(choices[0])
            if text:
                return text
        message = value.get("message")
        if isinstance(message, dict):
            text = extract_text(message)
            if text:
                return text
    return as_text(value)
