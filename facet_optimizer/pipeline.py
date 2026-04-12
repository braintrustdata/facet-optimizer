from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from .config import CreateDatasetConfig
from .dataset_rows import (
    assign_dataset_splits,
    bucket_counts,
    build_dataset_row,
    split_counts,
    upload_dataset_rows,
)
from .final_facets import attach_final_facet_values
from .ground_truth import GroundTruthLabeler
from .jsonl import write_json, write_jsonl
from .models import DatasetRow, ParsedFacetCall
from .parse_facet_call import parse_facet_calls
from .prompt_artifacts import write_prompt_artifact
from .source import (
    fetch_final_facets,
    fetch_source_spans,
    resolve_login_org_name,
    resolve_project_id,
)


def run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


Progress = Callable[[str], None]
LabelProgress = Callable[[int, int], None]


def noop_progress(_: str) -> None:
    return None


def select_balanced_calls(
    calls: list[ParsedFacetCall],
    *,
    positive_limit: int,
    negative_limit: int,
    explicit_roots: bool,
) -> list[ParsedFacetCall]:
    if explicit_roots:
        return calls

    selected: list[ParsedFacetCall] = []
    used_trace_facets: set[tuple[str, str]] = set()
    counts: Counter[tuple[str, str]] = Counter()
    limits = {"positive": positive_limit, "negative": negative_limit}
    for call in calls:
        key = (call.root_span_id, call.facet_name)
        if key in used_trace_facets:
            continue
        if call.weak_bucket not in limits:
            continue
        facet_bucket = (call.facet_name, call.weak_bucket)
        if counts[facet_bucket] >= limits[call.weak_bucket]:
            continue

        selected.append(call)
        used_trace_facets.add(key)
        counts[facet_bucket] += 1
    return selected


def create_facet_dataset(
    config: CreateDatasetConfig,
    *,
    progress: Progress | None = None,
    label_progress: LabelProgress | None = None,
) -> dict[str, object]:
    report = progress or noop_progress
    this_run_id = run_id()
    run_dir = config.output_root / this_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report(f"Run {this_run_id}: writing artifacts to {run_dir}")

    source_limit = config.limit
    if source_limit is None:
        source_limit = max(1, (config.positive_limit + config.negative_limit) * 5)

    source_query_org = config.source_org
    source_org = config.source_org or resolve_login_org_name(
        api_key=config.braintrust_api_key,
        app_url=config.app_url,
        org_name=None,
    )
    source_project_id = resolve_project_id(
        config.source_project,
        env_file=config.env_file,
        org_name=source_query_org,
    )
    if source_query_org:
        report(f"Using configured source org {source_query_org!r}")
    elif not source_org:
        report("Source trace permalinks will be omitted")

    if config.root_span_ids:
        report(
            "Fetching source facet spans for "
            f"{len(config.root_span_ids)} explicit root span id(s) "
            f"from {config.source_project!r}"
        )
    else:
        report(
            "Fetching source facet spans from "
            f"{config.source_project!r} with model {config.source_model!r} "
            f"(limit {source_limit})"
        )
    source_project_id, query, spans = fetch_source_spans(
        env_file=config.env_file,
        org_name=source_query_org,
        source_project_id=source_project_id,
        source_model=config.source_model,
        created_after_sql=config.created_after_sql,
        created_before_sql=config.created_before_sql,
        extra_where_sql=config.extra_where_sql,
        limit=source_limit,
        root_span_ids=config.root_span_ids,
    )
    (run_dir / "query.sql").write_text(query.rstrip() + "\n", encoding="utf-8")
    write_jsonl(run_dir / "candidates.jsonl", [span.__dict__ for span in spans])
    report(f"Fetched {len(spans)} candidate span(s)")

    root_span_ids = [span.root_span_id for span in spans]
    report(f"Fetching finalized facets for {len(set(root_span_ids))} root span id(s)")
    facets_queries, facets_by_root = fetch_final_facets(
        env_file=config.env_file,
        org_name=source_query_org,
        project_id=source_project_id,
        root_span_ids=root_span_ids,
    )
    (run_dir / "facets_query.sql").write_text(
        "\n\n-- next batch\n\n".join(query.rstrip() for query in facets_queries) + "\n",
        encoding="utf-8",
    )
    write_jsonl(
        run_dir / "final_facets.jsonl",
        [facets.__dict__ for facets in facets_by_root.values()],
    )
    report(f"Fetched finalized facets for {len(facets_by_root)} root span id(s)")

    parsed_calls: list[ParsedFacetCall] = []
    failed_rows: list[dict[str, str]] = []
    report("Parsing facet calls")
    for span in spans:
        try:
            parsed_calls.extend(parse_facet_calls(span, facet_name=config.facet_name))
        except Exception as exc:
            failed_rows.append({"source_id": span.id, "error": str(exc)})
    report(
        f"Parsed {len(parsed_calls)} facet call(s)"
        + (f"; {len(failed_rows)} parse failure(s)" if failed_rows else "")
    )
    parsed_calls = attach_final_facet_values(parsed_calls, facets_by_root)

    selected_calls = select_balanced_calls(
        parsed_calls,
        positive_limit=config.positive_limit,
        negative_limit=config.negative_limit,
        explicit_roots=bool(config.root_span_ids),
    )
    weak_counts = Counter(call.weak_bucket for call in selected_calls)
    per_facet_weak_counts = Counter(
        (call.facet_name, call.weak_bucket) for call in selected_calls
    )
    report(
        f"Selected {len(selected_calls)} call(s) for labeling "
        f"(per-facet weak buckets: {dict(sorted(per_facet_weak_counts.items()))})"
    )
    write_jsonl(
        run_dir / "parsed_calls.jsonl",
        [call.to_artifact() for call in selected_calls],
    )
    prompt_hashes = write_prompt_artifact(run_dir / "facet_prompt.yaml", selected_calls)
    report(f"Wrote parsed calls and prompt artifact ({len(prompt_hashes)} prompt hash(es))")

    labeler = GroundTruthLabeler(
        model=config.ground_truth_model,
        api_key=config.ground_truth_api_key,
        api_base=config.ground_truth_api_base,
    )
    dataset_rows_by_index: list[tuple[int, DatasetRow]] = []
    worker_count = min(config.concurrency, max(1, len(selected_calls)))
    report(
        f"Labeling {len(selected_calls)} call(s) with {config.ground_truth_model!r} "
        f"using {worker_count} worker(s)"
    )

    def label_and_build(index: int, call: ParsedFacetCall) -> tuple[int, DatasetRow]:
        ground_truth = labeler.label(call)
        return (
            index,
            build_dataset_row(
                call=call,
                ground_truth=ground_truth,
                source_project=config.source_project,
                source_project_id=source_project_id,
                source_org=source_org,
                source_model=config.source_model,
                app_url=config.app_url,
                prompt_hash=prompt_hashes.get(call.facet_name),
            ),
        )

    completed_labels = 0
    if selected_calls:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(label_and_build, index, call): call
                for index, call in enumerate(selected_calls)
            }
            for future in as_completed(futures):
                call = futures[future]
                try:
                    dataset_rows_by_index.append(future.result())
                except Exception as exc:
                    failed_rows.append(
                        {
                            "source_id": call.source_id,
                            "facet_name": call.facet_name,
                            "error": str(exc),
                        }
                    )
                completed_labels += 1
                if label_progress:
                    label_progress(completed_labels, len(selected_calls))

    dataset_rows = [
        row for _, row in sorted(dataset_rows_by_index, key=lambda item: item[0])
    ]
    dataset_rows = assign_dataset_splits(
        dataset_rows,
        validation_fraction=config.validation_fraction,
        seed=f"{config.source_project}:{config.target_project}:{config.dataset}",
    )

    write_jsonl(run_dir / "dataset_rows.jsonl", [row.to_dict() for row in dataset_rows])
    report(f"Wrote {len(dataset_rows)} dataset row(s)")

    uploaded_rows = 0
    if not config.dry_run and not config.skip_upload:
        report(
            f"Uploading {len(dataset_rows)} row(s) to "
            f"{config.target_project!r} / {config.dataset!r}"
        )
        uploaded_rows = upload_dataset_rows(
            rows=dataset_rows,
            project=config.target_project,
            dataset_name=config.dataset,
            api_key=config.braintrust_api_key,
            app_url=config.app_url,
            org_name=config.target_org,
        )
        report(f"Uploaded {uploaded_rows} row(s)")
    else:
        report("Skipping upload")

    summary: dict[str, object] = {
        "run_id": this_run_id,
        "run_dir": str(run_dir),
        "source_project": config.source_project,
        "source_project_id": source_project_id,
        "source_org": source_org,
        "target_project": config.target_project,
        "dataset": config.dataset,
        "source_model": config.source_model,
        "extra_where_sql": config.extra_where_sql,
        "ground_truth_model": config.ground_truth_model,
        "concurrency": config.concurrency,
        "validation_fraction": config.validation_fraction,
        "candidate_count": len(spans),
        "final_facets_count": len(facets_by_root),
        "parsed_call_count": len(parsed_calls),
        "selected_call_count": len(selected_calls),
        "dataset_row_count": len(dataset_rows),
        "uploaded_row_count": uploaded_rows,
        "weak_bucket_counts": dict(sorted(weak_counts.items())),
        "per_facet_weak_bucket_counts": {
            f"{facet}:{bucket}": count
            for (facet, bucket), count in sorted(per_facet_weak_counts.items())
        },
        "final_bucket_counts": bucket_counts(dataset_rows),
        "split_counts": split_counts(dataset_rows),
        "prompt_hashes": prompt_hashes,
        "failed_rows": failed_rows,
        "dry_run": config.dry_run,
        "skip_upload": config.skip_upload,
    }
    write_json(run_dir / "summary.json", summary)
    report(f"Wrote summary to {run_dir / 'summary.json'}")
    return summary
