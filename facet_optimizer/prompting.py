from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .common import DEFAULT_NONE_VALUE, PROMPT_PLACEHOLDER, normalize_for_match


def load_prompt_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object")
    if not isinstance(payload.get("messages"), list):
        raise ValueError(f"{path} is missing messages")
    return payload


def save_prompt_spec(path: Path, prompt_spec: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(
            prompt_spec,
            sort_keys=False,
            allow_unicode=False,
            width=100,
        )
    )


def render_messages(prompt_spec: dict[str, Any], *, preprocessed_text: str) -> tuple[list[dict[str, str]], list[list[dict[str, str]]]]:
    placeholder = "{{" + str(prompt_spec.get("placeholder", "preprocessed_text")) + "}}"
    messages = copy.deepcopy(prompt_spec["messages"])
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content.replace(placeholder, preprocessed_text)

    suffix_messages = copy.deepcopy(prompt_spec.get("suffix_messages") or [])
    for group in suffix_messages:
        for message in group:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content.replace(placeholder, preprocessed_text)

    return messages, suffix_messages


def none_value_from_spec(prompt_spec: dict[str, Any]) -> str:
    normalization = prompt_spec.get("normalization") or {}
    none_value = normalization.get("none_value")
    if isinstance(none_value, str) and none_value.strip():
        return none_value.strip()
    return DEFAULT_NONE_VALUE


def normalize_expected(value: Any, *, prompt_spec: dict[str, Any]) -> str:
    return normalize_for_match(value, none_value=none_value_from_spec(prompt_spec))


def is_none_like(value: Any, *, prompt_spec: dict[str, Any]) -> bool:
    return normalize_expected(value, prompt_spec=prompt_spec) == none_value_from_spec(prompt_spec)

