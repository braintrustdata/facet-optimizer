#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from facet_optimizer.common import (
    DEFAULT_CREATED_AFTER_SQL,
    DEFAULT_CREATED_BEFORE_SQL,
    DEFAULT_NEGATIVE_VALUE,
    batched,
    ensure_dir,
    facet_identifier,
    run_bt_sql,
    sql_string,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample positive and negative automation traces for a Braintrust facet."
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--facet-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--prefer-profile", action="store_true")
    parser.add_argument("--positive-limit", type=int, default=100)
    parser.add_argument("--negative-limit", type=int, default=100)
    parser.add_argument("--created-after-sql", default=DEFAULT_CREATED_AFTER_SQL)
    parser.add_argument("--created-before-sql", default=DEFAULT_CREATED_BEFORE_SQL)
    parser.add_argument("--negative-value", default=DEFAULT_NEGATIVE_VALUE)
    parser.add_argument("--automation-span-type", default="automation")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_root_query(
    *,
    project_id: str,
    facet_name: str,
    span_type: str,
    created_after_sql: str,
    created_before_sql: str,
    negative_value: str,
    limit: int,
    bucket: str,
) -> str:
    facet_expr = facet_identifier(facet_name)
    if bucket == "positive":
        facet_filter = f"{facet_expr} IS NOT NULL AND {facet_expr} != {sql_string(negative_value)}"
    elif bucket == "negative":
        facet_filter = f"{facet_expr} IS NULL OR {facet_expr} = {sql_string(negative_value)}"
    else:
        raise ValueError(f"Unsupported bucket: {bucket}")

    return f"""SELECT
  root_span_id,
  MAX(created) AS latest_created
FROM project_logs({sql_string(project_id)})
WHERE span_attributes.type = {sql_string(span_type)}
  AND ({facet_filter})
  AND created >= {created_after_sql}
  AND created < {created_before_sql}
GROUP BY root_span_id
ORDER BY MAX(created) DESC
LIMIT {limit}"""


def build_automation_query(
    *,
    project_id: str,
    facet_name: str,
    span_type: str,
    root_ids: list[str],
) -> str:
    facet_expr = facet_identifier(facet_name)
    in_clause = ", ".join(sql_string(root_id) for root_id in root_ids)
    return f"""SELECT
  id,
  root_span_id,
  created,
  {facet_expr} AS facet_value
FROM project_logs({sql_string(project_id)})
WHERE span_attributes.type = {sql_string(span_type)}
  AND root_span_id IN ({in_clause})
ORDER BY created DESC"""


def latest_automation_rows(
    *,
    project_id: str,
    facet_name: str,
    span_type: str,
    root_ids: list[str],
    batch_size: int,
    profile: str | None,
    prefer_profile: bool,
    dry_run: bool,
    queries_dir: Path,
) -> dict[str, dict[str, Any]]:
    latest_by_root: dict[str, dict[str, Any]] = {}
    for batch_index, root_batch in enumerate(batched(root_ids, batch_size), start=1):
        query = build_automation_query(
            project_id=project_id,
            facet_name=facet_name,
            span_type=span_type,
            root_ids=root_batch,
        )
        write_text(queries_dir / f"automation_rows_batch_{batch_index:03d}.sql", query)
        rows = run_bt_sql(
            query,
            profile=profile,
            prefer_profile=prefer_profile,
            dry_run=dry_run,
        )
        for row in rows:
            root_span_id = row.get("root_span_id")
            if not root_span_id or root_span_id in latest_by_root:
                continue
            latest_by_root[root_span_id] = row
    return latest_by_root


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    queries_dir = output_dir / "queries"
    ensure_dir(output_dir)
    ensure_dir(queries_dir)

    positive_query = build_root_query(
        project_id=args.project_id,
        facet_name=args.facet_name,
        span_type=args.automation_span_type,
        created_after_sql=args.created_after_sql,
        created_before_sql=args.created_before_sql,
        negative_value=args.negative_value,
        limit=args.positive_limit,
        bucket="positive",
    )
    negative_query = build_root_query(
        project_id=args.project_id,
        facet_name=args.facet_name,
        span_type=args.automation_span_type,
        created_after_sql=args.created_after_sql,
        created_before_sql=args.created_before_sql,
        negative_value=args.negative_value,
        limit=args.negative_limit,
        bucket="negative",
    )

    write_text(queries_dir / "positive_roots.sql", positive_query)
    write_text(queries_dir / "negative_roots.sql", negative_query)

    positive_rows = run_bt_sql(
        positive_query,
        profile=args.profile,
        prefer_profile=args.prefer_profile,
        dry_run=args.dry_run,
    )
    negative_rows = run_bt_sql(
        negative_query,
        profile=args.profile,
        prefer_profile=args.prefer_profile,
        dry_run=args.dry_run,
    )

    positive_root_ids = [row["root_span_id"] for row in positive_rows if row.get("root_span_id")]
    negative_root_ids = [row["root_span_id"] for row in negative_rows if row.get("root_span_id")]
    all_root_ids = positive_root_ids + negative_root_ids

    automation_by_root = latest_automation_rows(
        project_id=args.project_id,
        facet_name=args.facet_name,
        span_type=args.automation_span_type,
        root_ids=all_root_ids,
        batch_size=args.batch_size,
        profile=args.profile,
        prefer_profile=args.prefer_profile,
        dry_run=args.dry_run,
        queries_dir=queries_dir,
    )

    sampled_roots: list[dict[str, Any]] = []
    for bucket, rows in (("positive", positive_rows), ("negative", negative_rows)):
        for row in rows:
            root_span_id = row["root_span_id"]
            automation_row = automation_by_root.get(root_span_id, {})
            sampled_roots.append(
                {
                    "bucket": bucket,
                    "root_span_id": root_span_id,
                    "latest_created": row.get("latest_created"),
                    "facet_value": automation_row.get("facet_value"),
                    "automation_row_id": automation_row.get("id"),
                    "automation_row_created": automation_row.get("created"),
                }
            )

    write_json(output_dir / "positive_roots.json", [row for row in sampled_roots if row["bucket"] == "positive"])
    write_json(output_dir / "negative_roots.json", [row for row in sampled_roots if row["bucket"] == "negative"])
    write_json(output_dir / "sampled_roots.json", sampled_roots)
    write_text(output_dir / "positive_root_ids.txt", "\n".join(positive_root_ids))
    write_text(output_dir / "negative_root_ids.txt", "\n".join(negative_root_ids))

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "project_id": args.project_id,
        "facet_name": args.facet_name,
        "profile": args.profile,
        "prefer_profile": args.prefer_profile,
        "automation_span_type": args.automation_span_type,
        "negative_value": args.negative_value,
        "created_after_sql": args.created_after_sql,
        "created_before_sql": args.created_before_sql,
        "limits": {
            "positive": args.positive_limit,
            "negative": args.negative_limit,
        },
        "files": {
            "sampled_roots": "sampled_roots.json",
            "positive_roots": "positive_roots.json",
            "negative_roots": "negative_roots.json",
        },
    }
    write_json(output_dir / "sample_manifest.json", manifest)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sampled_roots": len(sampled_roots),
                "positive_roots": len(positive_root_ids),
                "negative_roots": len(negative_root_ids),
                "dry_run": args.dry_run,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)

