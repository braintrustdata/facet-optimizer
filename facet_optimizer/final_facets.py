from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from .models import FinalFacets, ParsedFacetCall, WeakBucket
from .text import extract_text, is_none_like

SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


def normalize_facet_key(value: Any) -> str:
    return SEPARATOR_RE.sub(" ", str(value).strip().lower()).strip()


def final_bucket_for_value(value: Any) -> WeakBucket:
    if value is None:
        return "unknown"
    text = extract_text(value)
    if not text:
        return "negative"
    return "negative" if is_none_like(text) else "positive"


def find_final_facet_value(facets: dict[str, Any], facet_name: str) -> Any:
    if facet_name in facets:
        return facets[facet_name]

    lowered = facet_name.lower()
    for key, value in facets.items():
        if key.lower() == lowered:
            return value

    normalized = normalize_facet_key(facet_name)
    for key, value in facets.items():
        if normalize_facet_key(key) == normalized:
            return value

    return None


def attach_final_facet_values(
    calls: list[ParsedFacetCall],
    facets_by_root: dict[str, FinalFacets],
) -> list[ParsedFacetCall]:
    resolved: list[ParsedFacetCall] = []
    for call in calls:
        final_facets = facets_by_root.get(call.root_span_id)
        value = (
            find_final_facet_value(final_facets.facets, call.facet_name)
            if final_facets
            else None
        )
        value_text = extract_text(value) if value is not None else None
        resolved.append(
            replace(
                call,
                source_facet_value=value_text,
                weak_bucket=final_bucket_for_value(value),
            )
        )
    return resolved
