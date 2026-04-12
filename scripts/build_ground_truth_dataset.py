#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from facet_optimizer.common import (
    DEFAULT_NONE_VALUE,
    batched,
    ensure_dir,
    read_json_or_jsonl,
    run_bt_sql,
    slugify,
    sql_string,
    write_json,
    write_jsonl,
    write_text,
)
from facet_optimizer.factoring import (
    PromptFactoringError,
    factor_prompt,
    normalize_source_span,
)
from facet_optimizer.prompting import save_prompt_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch source LLM spans for sampled facet traces, factor the prompt into a "
            "versioned YAML file, and build a ground-truth seed dataset."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--llm-model", default="brain-facet-1")
    parser.add_argument("--llm-span-type", default="llm")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--input-message-index", type=int)
    parser.add_argument("--none-value", default=DEFAULT_NONE_VALUE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_llm_query(
    *,
    project_id: str,
    llm_span_type: str,
    llm_model: str,
    root_ids: list[str],
) -> str:
    in_clause = ", ".join(sql_string(root_id) for root_id in root_ids)
    return f"""SELECT
  id,
  root_span_id,
  created,
  span_attributes,
  metadata,
  input,
  output,
  metrics
FROM project_logs({sql_string(project_id)})
WHERE span_attributes.type = {sql_string(llm_span_type)}
  AND metadata.model = {sql_string(llm_model)}
  AND root_span_id IN ({in_clause})
ORDER BY created ASC"""


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    sampled_roots_path = manifest_path.parent / manifest["files"]["sampled_roots"]
    sampled_roots = read_json_or_jsonl(sampled_roots_path)

    output_root = Path(args.output_root).expanduser().resolve()
    prompt_dir = output_root / "prompts"
    dataset_dir = output_root / "datasets"
    artifact_dir = output_root / "artifacts" / f"{slugify(manifest['facet_name'])}_{args.prompt_version}"
    query_dir = artifact_dir / "queries"
    ensure_dir(prompt_dir)
    ensure_dir(dataset_dir)
    ensure_dir(artifact_dir)
    ensure_dir(query_dir)

    roots_by_id = {row["root_span_id"]: row for row in sampled_roots if row.get("root_span_id")}
    root_ids = list(roots_by_id)

    all_rows: list[dict[str, Any]] = []
    for batch_index, root_batch in enumerate(batched(root_ids, args.batch_size), start=1):
        query = build_llm_query(
            project_id=manifest["project_id"],
            llm_span_type=args.llm_span_type,
            llm_model=args.llm_model,
            root_ids=root_batch,
        )
        write_text(query_dir / f"llm_spans_batch_{batch_index:03d}.sql", query)
        rows = run_bt_sql(
            query,
            profile=manifest.get("profile"),
            prefer_profile=bool(manifest.get("prefer_profile")),
            dry_run=args.dry_run,
        )
        all_rows.extend(rows)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "facet_name": manifest["facet_name"],
                    "sampled_roots": len(root_ids),
                    "query_batches": len(batched(root_ids, args.batch_size)),
                    "artifact_dir": str(artifact_dir),
                    "dry_run": True,
                }
            )
        )
        return 0

    rows_by_root: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        root_span_id = row.get("root_span_id")
        if not root_span_id:
            continue
        sampled_root = roots_by_id.get(root_span_id)
        if not sampled_root:
            continue
        exported_row = {
            "id": row.get("id"),
            "root_span_id": root_span_id,
            "created": row.get("created"),
            "span_attributes": row.get("span_attributes"),
            "metadata": row.get("metadata"),
            "input": row.get("input"),
            "output": row.get("output"),
            "metrics": row.get("metrics"),
            "sample_bucket": sampled_root.get("bucket"),
            "sampled_facet_name": manifest["facet_name"],
            "sampled_facet_value": sampled_root.get("facet_value"),
            "sampled_automation_row_id": sampled_root.get("automation_row_id"),
            "sampled_automation_row_created": sampled_root.get("automation_row_created"),
            "sampled_root_latest_created": sampled_root.get("latest_created"),
        }
        rows_by_root.setdefault(root_span_id, []).append(exported_row)

    selected_rows: list[dict[str, Any]] = []
    multi_span_roots: dict[str, int] = {}
    for root_id, rows in rows_by_root.items():
        if len(rows) > 1:
            multi_span_roots[root_id] = len(rows)
        selected_rows.append(rows[-1])

    selected_rows.sort(key=lambda row: (str(row.get("created")), str(row.get("root_span_id"))))

    normalized_spans = [
        normalize_source_span(row, none_value=args.none_value) for row in selected_rows
    ]
    prompt_spec, dataset_rows, summary = factor_prompt(
        normalized_spans,
        facet_name=manifest["facet_name"],
        prompt_version=args.prompt_version,
        project_id=manifest["project_id"],
        llm_model=args.llm_model,
        input_message_index=args.input_message_index,
        none_value=args.none_value,
    )

    prompt_filename = f"facet_{slugify(manifest['facet_name'])}_{args.prompt_version}.yaml"
    dataset_stem = f"facet_{slugify(manifest['facet_name'])}_{args.prompt_version}_seed"

    prompt_path = prompt_dir / prompt_filename
    dataset_json_path = dataset_dir / f"{dataset_stem}.json"
    dataset_jsonl_path = dataset_dir / f"{dataset_stem}.jsonl"
    raw_spans_path = artifact_dir / "source_llm_spans.json"
    summary_path = artifact_dir / "factoring_summary.json"

    for row in dataset_rows:
        metadata = row.setdefault("metadata", {})
        metadata["prompt_file"] = prompt_filename

    save_prompt_spec(prompt_path, prompt_spec)
    write_json(dataset_json_path, dataset_rows)
    write_jsonl(dataset_jsonl_path, dataset_rows)
    write_json(raw_spans_path, selected_rows)

    missing_roots = [root_id for root_id in root_ids if root_id not in rows_by_root]
    bucket_counts = Counter(row.get("sample_bucket") for row in selected_rows)

    summary_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "facet_name": manifest["facet_name"],
        "prompt_version": args.prompt_version,
        "llm_model": args.llm_model,
        "counts": {
            "sampled_roots": len(root_ids),
            "selected_source_spans": len(selected_rows),
            "missing_roots": len(missing_roots),
        },
        "bucket_counts": dict(bucket_counts),
        "missing_root_span_ids": missing_roots,
        "roots_with_multiple_llm_spans": multi_span_roots,
        "files": {
            "prompt": str(prompt_path.relative_to(output_root)),
            "dataset_json": str(dataset_json_path.relative_to(output_root)),
            "dataset_jsonl": str(dataset_jsonl_path.relative_to(output_root)),
            "source_llm_spans": str(raw_spans_path.relative_to(output_root)),
        },
        "factoring": summary,
    }
    write_json(summary_path, summary_payload)

    print(
        json.dumps(
            {
                "prompt": str(prompt_path),
                "dataset": str(dataset_jsonl_path),
                "selected_source_spans": len(selected_rows),
                "missing_roots": len(missing_roots),
                "dry_run": args.dry_run,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, PromptFactoringError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
