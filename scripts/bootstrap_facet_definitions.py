#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from facet_optimizer.facet_definitions import latest_prompt_path, load_facet_definitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a run's facet_prompt.yaml to the local current eval prompt."
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".local/facet-optimizer"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".local/facet-optimizer/facet_definitions.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_dir:
        source = args.run_dir.expanduser() / "facet_prompt.yaml"
    else:
        source = latest_prompt_path(args.output_root)
        if source is None:
            raise SystemExit("error: no facet_prompt.yaml found under output root")

    if not source.exists():
        raise SystemExit(f"error: {source} does not exist")

    definitions = load_facet_definitions(source)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)

    summary = {
        "source": str(source),
        "output": str(output),
        "facets": sorted(definitions),
        "prompt_hashes": {
            name: definition.prompt_sha256
            for name, definition in sorted(definitions.items())
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nFACET_OPTIMIZER_PROMPT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
