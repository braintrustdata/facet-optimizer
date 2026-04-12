from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .models import ParsedFacetCall


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def templated_messages(call: ParsedFacetCall) -> list[dict[str, str]]:
    if not call.base_messages:
        return [
            {
                "role": "user",
                "content": "Here is the data to analyze:\n\n{{preprocessed_text}}",
            }
        ]
    messages: list[dict[str, str]] = []
    for message in call.base_messages:
        content = message["content"]
        if call.preprocessed_text:
            content = content.replace(call.preprocessed_text, "{{preprocessed_text}}")
        messages.append({"role": message["role"], "content": content})
    return messages


def artifact_for_call(call: ParsedFacetCall) -> dict[str, object]:
    return {
        "facet_name": call.facet_name,
        "prompt_sha256": prompt_sha256(call.facet_prompt),
        "placeholder": "preprocessed_text",
        "messages": templated_messages(call),
        "suffix_messages": [[{"role": "user", "content": call.facet_prompt}]],
    }


def write_prompt_artifact(path: Path, calls: list[ParsedFacetCall]) -> dict[str, str]:
    calls_by_facet: dict[str, ParsedFacetCall] = {}
    for call in calls:
        calls_by_facet.setdefault(call.facet_name, call)

    if not calls_by_facet:
        payload = {"schema_version": 1, "facets": []}
    elif len(calls_by_facet) == 1:
        call = next(iter(calls_by_facet.values()))
        payload = {
            "schema_version": 1,
            **artifact_for_call(call),
        }
    else:
        payload = {
            "schema_version": 1,
            "facets": [
                artifact_for_call(call)
                for _, call in sorted(calls_by_facet.items())
            ],
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {
        facet_name: prompt_sha256(call.facet_prompt)
        for facet_name, call in calls_by_facet.items()
    }
