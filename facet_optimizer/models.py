from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


JsonDict = dict[str, Any]
Message = dict[str, str]
WeakBucket = Literal["positive", "negative", "unknown"]


@dataclass(frozen=True)
class SourceSpan:
    id: str
    root_span_id: str
    span_id: str
    created: str | None
    input: Any
    output: Any
    metadata: JsonDict
    span_attributes: JsonDict


@dataclass(frozen=True)
class FinalFacets:
    row_id: str
    root_span_id: str
    span_id: str
    created: str | None
    facets: JsonDict


@dataclass(frozen=True)
class ParsedFacetCall:
    source_id: str
    root_span_id: str
    span_id: str
    source_created: str | None
    facet_name: str
    facet_prompt: str
    preprocessed_text: str
    production_output: str | None
    source_facet_value: str | None
    weak_bucket: WeakBucket
    base_messages: list[Message] = field(default_factory=list)

    def to_artifact(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class GroundTruthResult:
    expected: str
    model: str
    generated_at: str
    raw_output: str | None = None


@dataclass(frozen=True)
class DatasetRow:
    id: str
    input: JsonDict
    expected: str
    metadata: JsonDict
    tags: list[str]

    def to_dict(self) -> JsonDict:
        return asdict(self)
