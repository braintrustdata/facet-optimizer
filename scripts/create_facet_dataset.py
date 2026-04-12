#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from facet_optimizer.config import add_common_args, load_config
from facet_optimizer.pipeline import create_facet_dataset


class StderrProgress:
    def __init__(self) -> None:
        self._bar_active = False

    def log(self, message: str) -> None:
        if self._bar_active:
            print(file=sys.stderr, flush=True)
            self._bar_active = False
        print(message, file=sys.stderr, flush=True)

    def label_bar(self, done: int, total: int) -> None:
        width = 32
        filled = width if total <= 0 else int(width * done / total)
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\rLabeling [{bar}] {done}/{total}",
            end="",
            file=sys.stderr,
            flush=True,
        )
        self._bar_active = done < total
        if done >= total:
            print(file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a ground-truth facet dataset from production Braintrust facet LLM spans."
    )
    add_common_args(parser)
    return parser.parse_args()


def main() -> int:
    progress = StderrProgress()
    try:
        config = load_config(parse_args())
        summary = create_facet_dataset(
            config,
            progress=progress.log,
            label_progress=progress.label_bar,
        )
    except Exception as exc:
        if progress._bar_active:
            print(file=sys.stderr, flush=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
