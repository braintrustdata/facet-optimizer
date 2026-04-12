from __future__ import annotations

import re
from typing import Any

from .models import Message, ParsedFacetCall, SourceSpan, WeakBucket
from .text import extract_text, is_none_like

DATA_MARKERS = (
    "Here is the data to analyze:",
    "Here is the data to analyse:",
)


def _message(role: Any, content: Any) -> Message | None:
    if not isinstance(role, str):
        return None
    text = extract_text(content)
    if not text:
        return None
    return {"role": role, "content": text}


def _parse_messages(value: Any) -> list[Message]:
    if not isinstance(value, list):
        return []
    messages = []
    for item in value:
        if not isinstance(item, dict):
            continue
        msg = _message(item.get("role"), item.get("content"))
        if msg:
            messages.append(msg)
    return messages


def _parse_suffix_groups(value: Any) -> list[list[Message]]:
    if not isinstance(value, list):
        return []
    groups = []
    for group in value:
        parsed = _parse_messages(group)
        if parsed:
            groups.append(parsed)
    return groups


def _input_messages(input_value: Any) -> list[Message]:
    if isinstance(input_value, list):
        return _parse_messages(input_value)
    if isinstance(input_value, dict):
        messages = input_value.get("messages")
        if messages is None:
            messages = input_value.get("input")
        return _parse_messages(messages)
    return []


def _suffix_groups(input_value: Any, metadata: dict[str, Any]) -> list[list[Message]]:
    sources = []
    if isinstance(input_value, dict):
        sources.append(input_value)
    sources.append(metadata)
    for source in sources:
        for key in ("suffix_messages", "suffixMessages"):
            groups = _parse_suffix_groups(source.get(key))
            if groups:
                return groups
    return []


def _extract_after_marker(text: str) -> str | None:
    for marker in DATA_MARKERS:
        if marker in text:
            return text.split(marker, 1)[1].strip()
    return None


def extract_preprocessed_text(input_value: Any, messages: list[Message]) -> str:
    if isinstance(input_value, dict):
        for key in ("preprocessed_text", "preprocessedText", "user_data", "userData"):
            text = extract_text(input_value.get(key))
            if text:
                return text
    for msg in messages:
        text = _extract_after_marker(msg["content"])
        if text:
            return text
    if messages:
        return messages[-1]["content"].strip()
    return extract_text(input_value)


def _prompt_from_group(group: list[Message]) -> str:
    return "\n\n".join(msg["content"] for msg in group).strip()


def _fallback_prompt(messages: list[Message], preprocessed_text: str) -> str:
    candidates = []
    for msg in messages:
        content = msg["content"].strip()
        if not content or content == preprocessed_text:
            continue
        if _extract_after_marker(content) == preprocessed_text:
            continue
        candidates.append(content)
    return candidates[-1] if candidates else ""


def _metadata_facet_name(metadata: dict[str, Any], fallback: str | None) -> str | None:
    if fallback:
        return fallback
    for key in ("facet_name", "facet", "sampled_facet_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def infer_facet_name(prompt: str, metadata: dict[str, Any], fallback: str | None) -> str | None:
    explicit = _metadata_facet_name(metadata, fallback)
    if explicit:
        return explicit
    match = re.search(r"extracting an? [\"']?([^\"'\n]+?)[\"']? facet", prompt, re.I)
    if match:
        return re.sub(r"\s+", "-", match.group(1).strip().lower())
    lower = prompt.lower()
    if "no response" in lower and "leaked reasoning" in lower:
        return "issues"
    if all(term in lower for term in ("negative", "neutral", "positive", "mixed")):
        return "sentiment"
    if "user wants to" in lower or "overall request or goal" in lower:
        return "task"
    return None


def weak_bucket_for_output(output: Any) -> WeakBucket:
    text = extract_text(output)
    if not text:
        return "negative"
    return "negative" if is_none_like(text) else "positive"


def parse_facet_calls(span: SourceSpan, *, facet_name: str | None = None) -> list[ParsedFacetCall]:
    messages = _input_messages(span.input)
    suffix_groups = _suffix_groups(span.input, span.metadata)
    preprocessed_text = extract_preprocessed_text(span.input, messages)
    production_output = extract_text(span.output) or None
    weak_bucket = weak_bucket_for_output(production_output)

    prompt_groups = suffix_groups or [[{"role": "user", "content": _fallback_prompt(messages, preprocessed_text)}]]
    parsed: list[ParsedFacetCall] = []
    for index, group in enumerate(prompt_groups):
        facet_prompt = _prompt_from_group(group)
        if not facet_prompt:
            continue
        resolved_name = infer_facet_name(facet_prompt, span.metadata, facet_name)
        if not resolved_name:
            resolved_name = f"facet-{index + 1}" if len(prompt_groups) > 1 else "facet"
        parsed.append(
            ParsedFacetCall(
                source_id=span.id,
                root_span_id=span.root_span_id,
                span_id=span.span_id,
                source_created=span.created,
                facet_name=resolved_name,
                facet_prompt=facet_prompt,
                preprocessed_text=preprocessed_text,
                production_output=production_output,
                source_facet_value=None,
                weak_bucket=weak_bucket,
                base_messages=messages,
            )
        )
    return parsed
