from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
from typing import Iterable
from urllib.parse import quote, urlencode

from braintrust import init_dataset
from braintrust.logger import BraintrustState

from .models import DatasetRow, GroundTruthResult, ParsedFacetCall
from .text import is_none_like


def final_bucket(expected: str) -> str:
    return "negative" if is_none_like(expected) else "positive"


def split_key(row: DatasetRow, *, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{row.id}".encode("utf-8")).hexdigest()


def assign_dataset_splits(
    rows: list[DatasetRow],
    *,
    validation_fraction: float,
    seed: str,
) -> list[DatasetRow]:
    groups: dict[tuple[str, str], list[DatasetRow]] = {}
    for row in rows:
        facet_name = str(row.input.get("facet_name") or "")
        groups.setdefault((facet_name, final_bucket(row.expected)), []).append(row)

    validation_ids: set[str] = set()
    if validation_fraction > 0:
        for group_rows in groups.values():
            if len(group_rows) <= 1:
                continue
            validation_count = round(len(group_rows) * validation_fraction)
            validation_count = max(1, validation_count)
            validation_count = min(validation_count, len(group_rows) - 1)
            validation_rows = sorted(
                group_rows,
                key=lambda row: split_key(row, seed=seed),
            )[:validation_count]
            validation_ids.update(row.id for row in validation_rows)

    split_rows: list[DatasetRow] = []
    for row in rows:
        split = "validation" if row.id in validation_ids else "train"
        metadata = dict(row.metadata)
        metadata["split"] = split
        tags = list(row.tags)
        if split not in tags:
            tags.append(split)
        split_rows.append(replace(row, metadata=metadata, tags=tags))
    return split_rows


def build_dataset_row(
    *,
    call: ParsedFacetCall,
    ground_truth: GroundTruthResult,
    source_project: str,
    source_project_id: str,
    source_org: str | None,
    source_model: str,
    app_url: str | None,
    prompt_hash: str | None,
) -> DatasetRow:
    bucket = final_bucket(ground_truth.expected)
    return DatasetRow(
        id=f"{call.source_id}:{call.facet_name}",
        input={
            "facet_name": call.facet_name,
            "preprocessed_text": call.preprocessed_text,
        },
        expected=ground_truth.expected,
        metadata={
            "source_project": source_project,
            "source_project_id": source_project_id,
            "source_org": source_org,
            "source_model": source_model,
            "source_root_span_id": call.root_span_id,
            "source_span_id": call.span_id,
            "source_row_id": call.source_id,
            "source_created": call.source_created,
            "source_weak_bucket": call.weak_bucket,
            "source_facet_value": call.source_facet_value,
            "source_trace_permalink": trace_permalink(
                app_url=app_url,
                org_name=source_org,
                project_id=source_project_id,
                row_id=call.source_id,
                project=source_project,
                root_span_id=call.root_span_id,
                span_id=call.span_id,
            ),
            "production_output": call.production_output,
            "facet_prompt_sha256": prompt_hash,
            "ground_truth_model": ground_truth.model,
            "ground_truth_generated_at": ground_truth.generated_at,
        },
        tags=["facet-optimizer", call.facet_name, bucket],
    )


def trace_permalink(
    *,
    app_url: str | None,
    org_name: str | None,
    project_id: str | None,
    row_id: str | None,
    project: str,
    root_span_id: str,
    span_id: str,
) -> str | None:
    if not app_url or not org_name:
        return None
    base = app_url.rstrip("/")
    if project_id and row_id:
        path = f"/app/{quote(org_name, safe='')}/object"
        query = urlencode(
            {
                "object_type": "project_logs",
                "object_id": project_id,
                "id": row_id,
            }
        )
        return f"{base}{path}?{query}"
    path = f"/app/{quote(org_name, safe='')}/p/{quote(project, safe='')}/logs"
    query = urlencode({"r": root_span_id, "s": span_id})
    return f"{base}{path}?{query}"


def upload_dataset_rows(
    *,
    rows: Iterable[DatasetRow],
    project: str,
    dataset_name: str,
    api_key: str,
    app_url: str | None,
    org_name: str | None,
) -> int:
    state = BraintrustState()
    dataset = init_dataset(
        project=project,
        name=dataset_name,
        api_key=api_key,
        app_url=app_url,
        org_name=org_name,
        use_output=False,
        state=state,
    )
    count = 0
    for row in rows:
        dataset.insert(
            id=row.id,
            input=row.input,
            expected=row.expected,
            metadata=row.metadata,
            tags=row.tags,
        )
        count += 1
    dataset.flush()
    return count


def bucket_counts(rows: Iterable[DatasetRow]) -> dict[str, int]:
    counts = Counter(final_bucket(row.expected) for row in rows)
    return dict(sorted(counts.items()))


def split_counts(rows: Iterable[DatasetRow]) -> dict[str, int]:
    counts = Counter(str(row.metadata.get("split") or "unspecified") for row in rows)
    return dict(sorted(counts.items()))
