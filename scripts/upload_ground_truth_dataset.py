#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from braintrust import init_dataset
from dotenv import load_dotenv

from facet_optimizer.common import read_json_or_jsonl

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local ground-truth JSON or JSONL file into a Braintrust dataset."
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--app-url", default=os.getenv("BRAINTRUST_APP_URL"))
    parser.add_argument("--api-key", default=os.getenv("BRAINTRUST_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--insert-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("BRAINTRUST_API_KEY is required")

    input_path = Path(args.input_file).expanduser().resolve()
    rows = read_json_or_jsonl(input_path)

    dataset = init_dataset(
        project=args.project,
        name=args.dataset,
        app_url=args.app_url,
        api_key=args.api_key,
        use_output=False,
    )

    existing_ids: set[str] = set()
    if not args.insert_only:
        for row in dataset.fetch(batch_size=args.batch_size):
            row_id = row.get("id")
            if isinstance(row_id, str) and row_id:
                existing_ids.add(row_id)

    inserted = 0
    updated = 0
    skipped = 0
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            skipped += 1
            continue

        payload = {
            "id": row_id,
            "input": row.get("input"),
            "expected": row.get("expected"),
            "tags": row.get("tags"),
            "metadata": row.get("metadata"),
        }

        if row_id in existing_ids:
            dataset.update(**payload)
            updated += 1
        else:
            dataset.insert(**payload)
            inserted += 1

    dataset.flush()
    dataset.close()

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "project": args.project,
                "inserted": inserted,
                "updated": updated,
                "skipped": skipped,
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

