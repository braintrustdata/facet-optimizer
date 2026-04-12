from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .common import (
    DEFAULT_NONE_VALUE,
    PROMPT_PLACEHOLDER,
    as_text,
    canonicalize_none_like,
    extract_output_text,
    stable_row_id,
)


class PromptFactoringError(RuntimeError):
    pass


@dataclass
class NormalizedSpan:
    root_span_id: str
    span_id: str
    created: str
    sample_bucket: str | None
    sampled_facet_value: str
    source_output_text: str
    messages: list[dict[str, str]]
    suffix_messages: list[list[dict[str, str]]]
    metadata: dict[str, Any]


def _normalize_message(message: Any, *, path: str) -> dict[str, str]:
    if not isinstance(message, dict):
        raise PromptFactoringError(f"{path} must be an object")
    role = as_text(message.get("role"))
    content = message.get("content")
    if not role:
        raise PromptFactoringError(f"{path}.role is required")
    if not isinstance(content, str):
        raise PromptFactoringError(f"{path}.content must be a string")
    return {"role": role, "content": content}


def _normalize_message_group(value: Any, *, path: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise PromptFactoringError(f"{path} must be a list")
    return [_normalize_message(item, path=f"{path}[{index}]") for index, item in enumerate(value)]


def _normalize_suffix_messages(value: Any, *, path: str) -> list[list[dict[str, str]]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PromptFactoringError(f"{path} must be a list")
    return [_normalize_message_group(group, path=f"{path}[{index}]") for index, group in enumerate(value)]


def normalize_source_span(
    row: dict[str, Any],
    *,
    none_value: str = DEFAULT_NONE_VALUE,
) -> NormalizedSpan:
    input_value = row.get("input")
    metadata = dict(row.get("metadata") or {})
    if isinstance(input_value, dict):
        messages_value = input_value.get("messages")
        suffix_value = input_value.get("suffix_messages", metadata.get("suffix_messages"))
    else:
        messages_value = input_value
        suffix_value = metadata.get("suffix_messages")

    messages = _normalize_message_group(messages_value, path="input.messages")
    suffix_messages = _normalize_suffix_messages(suffix_value, path="suffix_messages")

    return NormalizedSpan(
        root_span_id=as_text(row.get("root_span_id")),
        span_id=as_text(row.get("id")),
        created=as_text(row.get("created")),
        sample_bucket=as_text(row.get("sample_bucket")) or None,
        sampled_facet_value=canonicalize_none_like(
            row.get("sampled_facet_value"),
            none_value=none_value,
        ),
        source_output_text=extract_output_text(row.get("output")),
        messages=messages,
        suffix_messages=suffix_messages,
        metadata=metadata,
    )


def _serialized_suffix_groups(groups: list[list[dict[str, str]]]) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        tuple((message["role"], message["content"]) for message in group)
        for group in groups
    )


def _longest_common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        while prefix and not value.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def _longest_common_suffix(values: list[str], *, prefix: str) -> str:
    if not values:
        return ""
    suffix = values[0][len(prefix) :]
    for value in values[1:]:
        candidate = value[len(prefix) :]
        while suffix and not candidate.endswith(suffix):
            suffix = suffix[1:]
    return suffix


def _extract_middle(value: str, *, prefix: str, suffix: str) -> str:
    if not value.startswith(prefix):
        raise PromptFactoringError("Prompt factoring failed: message did not match inferred prefix")
    if suffix and not value.endswith(suffix):
        raise PromptFactoringError("Prompt factoring failed: message did not match inferred suffix")
    end = len(value) - len(suffix) if suffix else len(value)
    return value[len(prefix) : end]


def factor_prompt(
    spans: list[NormalizedSpan],
    *,
    facet_name: str,
    prompt_version: str,
    project_id: str,
    llm_model: str,
    input_message_index: int | None = None,
    none_value: str = DEFAULT_NONE_VALUE,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not spans:
        raise PromptFactoringError("No source spans were provided")

    first = spans[0]
    message_count = len(first.messages)
    role_signature = [message["role"] for message in first.messages]
    suffix_signature = _serialized_suffix_groups(first.suffix_messages)

    for span in spans[1:]:
        if len(span.messages) != message_count:
            raise PromptFactoringError("Source spans do not share a stable message count")
        if [message["role"] for message in span.messages] != role_signature:
            raise PromptFactoringError("Source spans do not share a stable message role layout")
        if _serialized_suffix_groups(span.suffix_messages) != suffix_signature:
            raise PromptFactoringError("Source spans do not share a stable suffix_messages layout")

    varying_indices = [
        index
        for index in range(message_count)
        if len({span.messages[index]["content"] for span in spans}) > 1
    ]

    if input_message_index is None:
        if len(varying_indices) != 1:
            raise PromptFactoringError(
                "Could not infer a single varying message. "
                "Pass --input-message-index to override."
            )
        input_message_index = varying_indices[0]
    elif input_message_index < 0 or input_message_index >= message_count:
        raise PromptFactoringError(
            f"--input-message-index must be between 0 and {message_count - 1}"
        )

    varying_contents = [span.messages[input_message_index]["content"] for span in spans]
    prefix = _longest_common_prefix(varying_contents)
    suffix = _longest_common_suffix(varying_contents, prefix=prefix)

    template_messages = copy.deepcopy(first.messages)
    template_messages[input_message_index]["content"] = (
        prefix + PROMPT_PLACEHOLDER + suffix
    )

    rows: list[dict[str, Any]] = []
    positive_count = 0
    negative_count = 0
    source_output_counter: Counter[str] = Counter()
    for span in spans:
        preprocessed_text = _extract_middle(
            span.messages[input_message_index]["content"],
            prefix=prefix,
            suffix=suffix,
        )
        rendered = template_messages[input_message_index]["content"].replace(
            PROMPT_PLACEHOLDER,
            preprocessed_text,
        )
        if rendered != span.messages[input_message_index]["content"]:
            raise PromptFactoringError("Round-trip prompt rendering failed")

        if span.sample_bucket == "positive":
            positive_count += 1
        elif span.sample_bucket == "negative":
            negative_count += 1

        if span.source_output_text:
            source_output_counter[span.source_output_text] += 1

        row_id = stable_row_id(facet_name, span.root_span_id, prompt_version)
        rows.append(
            {
                "id": row_id,
                "input": {
                    "facet_name": facet_name,
                    "preprocessed_text": preprocessed_text,
                },
                "expected": span.sampled_facet_value or none_value,
                "tags": [f"facet:{facet_name}", f"bucket:{span.sample_bucket or 'unknown'}"],
                "metadata": {
                    "facet_name": facet_name,
                    "prompt_version": prompt_version,
                    "marked_positive": span.sample_bucket == "positive",
                    "source_sample_bucket": span.sample_bucket,
                    "source_root_span_id": span.root_span_id,
                    "source_llm_span_id": span.span_id,
                    "source_span_created": span.created,
                    "seed_expected": span.sampled_facet_value or none_value,
                    "source_output_text": span.source_output_text,
                },
            }
        )

    prompt_spec = {
        "schema_version": 1,
        "facet_name": facet_name,
        "prompt_version": prompt_version,
        "placeholder": "preprocessed_text",
        "source": {
            "project_id": project_id,
            "llm_model": llm_model,
            "sampled_span_count": len(spans),
        },
        "messages": template_messages,
        "suffix_messages": copy.deepcopy(first.suffix_messages),
        "normalization": {
            "none_value": none_value,
            "none_like_values": sorted({"NONE", "none", "no_match", "no match", "null"}),
        },
        "factoring": {
            "input_message_index": input_message_index,
            "varying_message_indices": varying_indices,
            "content_prefix": prefix,
            "content_suffix": suffix,
        },
    }

    summary = {
        "facet_name": facet_name,
        "prompt_version": prompt_version,
        "rows": len(rows),
        "positive_rows": positive_count,
        "negative_rows": negative_count,
        "message_count": message_count,
        "input_message_index": input_message_index,
        "varying_message_indices": varying_indices,
        "prefix_length": len(prefix),
        "suffix_length": len(suffix),
        "top_source_outputs": source_output_counter.most_common(5),
    }

    return prompt_spec, rows, summary

