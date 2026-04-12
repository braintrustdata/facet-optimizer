from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CREATED_AFTER_SQL = "NOW() - INTERVAL 7 DAY"
DEFAULT_CREATED_BEFORE_SQL = "NOW() - INTERVAL 2 HOUR"
DEFAULT_NEGATIVE_VALUE = "no_match"
DEFAULT_NONE_VALUE = "NONE"
PROMPT_PLACEHOLDER = "{{preprocessed_text}}"

NONE_LIKE_VALUES = {
    "",
    "none",
    "no match",
    "no_match",
    "null",
    "n/a",
    "na",
}

WHITESPACE_RE = re.compile(r"\s+")
SLUG_RE = re.compile(r"[^a-z0-9]+")
THINKING_TAGS_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open() as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return payload


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = SLUG_RE.sub("-", lowered).strip("-")
    return slug or "facet"


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def facet_identifier(facet_name: str) -> str:
    return f"facets.`{facet_name.replace('`', '``')}`"


def run_bt_sql(
    query: str,
    *,
    profile: str | None,
    prefer_profile: bool,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    cmd = ["bt", "sql", "--json", "--non-interactive"]
    if prefer_profile:
        cmd.append("--prefer-profile")
    if profile:
        cmd.extend(["--profile", profile])
    cmd.append(query)

    if dry_run:
        print(query)
        return []

    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "bt sql failed")

    payload = json.loads(completed.stdout)
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise RuntimeError("Unexpected bt sql JSON payload")


def strip_thinking_tags(text: str) -> str:
    return THINKING_TAGS_RE.sub("", text).strip()


def extract_output_text(output: Any) -> str:
    if isinstance(output, str):
        return strip_thinking_tags(output)
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return strip_thinking_tags(content)
    return ""


def canonicalize_none_like(value: Any, *, none_value: str = DEFAULT_NONE_VALUE) -> str:
    text = as_text(value)
    if text.lower() in NONE_LIKE_VALUES:
        return none_value
    return text


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value).strip()


def normalize_for_match(value: Any, *, none_value: str = DEFAULT_NONE_VALUE) -> str:
    text = canonicalize_none_like(value, none_value=none_value)
    return WHITESPACE_RE.sub(" ", text).strip()


def stable_row_id(facet_name: str, root_span_id: str, prompt_version: str) -> str:
    payload = f"{facet_name}|{root_span_id}|{prompt_version}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

