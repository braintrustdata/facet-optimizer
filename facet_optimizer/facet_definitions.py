from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import Message
from .prompt_artifacts import prompt_sha256


DEFAULT_NONE_PATTERN = r"(?i)^\s*(?:none|no[_ ]?match|null|n/a|na|skipped)\b"
PLACEHOLDER = "{{preprocessed_text}}"


@dataclass(frozen=True)
class FacetDefinition:
    facet_name: str
    messages: list[Message]
    suffix_messages: list[list[Message]]
    prompt_sha256: str
    prompt_text: str
    placeholder: str = "preprocessed_text"
    no_match_pattern: str = DEFAULT_NONE_PATTERN


def _as_message(value: Any) -> Message:
    if not isinstance(value, dict):
        raise ValueError(f"Message must be an object, got {type(value).__name__}")
    role = value.get("role")
    content = value.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise ValueError("Message must include string role and content")
    return {"role": role, "content": content}


def _as_suffix_messages(value: Any) -> list[list[Message]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("suffix_messages must be a list")
    groups: list[list[Message]] = []
    for group in value:
        if not isinstance(group, list):
            raise ValueError("Each suffix_messages group must be a list")
        groups.append([_as_message(message) for message in group])
    return groups


def _prompt_text_from_suffix_messages(groups: list[list[Message]]) -> str:
    if not groups:
        return ""
    return "\n\n".join(message["content"] for group in groups for message in group)


def _definition_from_raw(raw: dict[str, Any]) -> FacetDefinition:
    facet_name = raw.get("facet_name")
    if not isinstance(facet_name, str) or not facet_name.strip():
        raise ValueError("Facet definition is missing facet_name")

    messages_raw = raw.get("messages")
    if not isinstance(messages_raw, list):
        raise ValueError(f"Facet {facet_name!r} is missing messages")
    messages = [_as_message(message) for message in messages_raw]
    suffix_messages = _as_suffix_messages(raw.get("suffix_messages"))
    prompt_text = _prompt_text_from_suffix_messages(suffix_messages)
    computed_hash = prompt_sha256(prompt_text) if prompt_text else ""

    return FacetDefinition(
        facet_name=facet_name.strip(),
        messages=messages,
        suffix_messages=suffix_messages,
        prompt_sha256=computed_hash,
        prompt_text=prompt_text,
        placeholder=str(raw.get("placeholder") or "preprocessed_text"),
        no_match_pattern=str(raw.get("no_match_pattern") or DEFAULT_NONE_PATTERN),
    )


def load_facet_definitions(path: str | Path) -> dict[str, FacetDefinition]:
    prompt_path = Path(path).expanduser()
    payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{prompt_path} must contain a YAML object")

    raw_facets = payload.get("facets")
    if isinstance(raw_facets, list):
        definitions = [_definition_from_raw(raw) for raw in raw_facets]
    else:
        definitions = [_definition_from_raw(payload)]

    by_name = {definition.facet_name.lower(): definition for definition in definitions}
    if len(by_name) != len(definitions):
        raise ValueError(f"{prompt_path} contains duplicate facet names")
    return by_name


def render_messages(
    definition: FacetDefinition,
    *,
    preprocessed_text: str,
) -> list[Message]:
    rendered: list[Message] = []
    for message in definition.messages:
        rendered.append(
            {
                "role": message["role"],
                "content": message["content"].replace(PLACEHOLDER, preprocessed_text),
            }
        )
    return rendered


def latest_prompt_path(output_root: str | Path = ".local/facet-optimizer") -> Path | None:
    root = Path(output_root).expanduser()
    if not root.exists():
        return None
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "facet_prompt.yaml").exists()
    )
    if not candidates:
        return None
    return candidates[-1] / "facet_prompt.yaml"
