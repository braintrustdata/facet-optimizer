#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from braintrust.logger import api_conn, app_conn, login
from dotenv import load_dotenv

from facet_optimizer.facet_definitions import latest_prompt_path, load_facet_definitions


SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    slug = SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or "facet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish local facet prompt definitions as Braintrust facet functions."
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--project")
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--facet",
        action="append",
        default=[],
        help="YAML facet name to publish. Repeat to publish a subset.",
    )
    parser.add_argument(
        "--facet-name",
        help="Published name/slug for a single selected facet.",
    )
    parser.add_argument("--slug")
    parser.add_argument("--slug-prefix", default="facet-optimizer")
    parser.add_argument(
        "--if-exists",
        choices=["error", "replace", "ignore"],
        default="replace",
    )
    parser.add_argument("--target-org")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_prompt_path(args: argparse.Namespace) -> Path:
    if args.prompt:
        return args.prompt.expanduser()
    env_prompt = os.getenv("FACET_OPTIMIZER_PROMPT")
    if env_prompt:
        return Path(env_prompt).expanduser()
    latest = latest_prompt_path(
        os.getenv("FACET_OPTIMIZER_OUTPUT_ROOT", ".local/facet-optimizer")
    )
    if latest is None:
        raise ValueError("Set --prompt or run create_facet_dataset.py first")
    return latest


def resolve_project(args: argparse.Namespace) -> str:
    project = (
        args.project
        or os.getenv("FACET_OPTIMIZER_TARGET_PROJECT")
        or os.getenv("FACET_OPTIMIZER_PROJECT")
        or "Facet Optimizer"
    )
    return project.strip()


def selected_definitions(args: argparse.Namespace, prompt_path: Path):
    definitions = load_facet_definitions(prompt_path)
    requested = {name.strip().lower() for name in args.facet if name.strip()}
    if requested:
        missing = requested - set(definitions)
        if missing:
            raise ValueError(f"Unknown facet(s): {', '.join(sorted(missing))}")
        definitions = {
            name: definition
            for name, definition in definitions.items()
            if name in requested
        }
    if len(definitions) != 1 and (args.facet_name or args.slug):
        raise ValueError("--facet-name and --slug require exactly one selected facet")
    return definitions


def project_id_for_name(project: str) -> str:
    response = app_conn().post_json("api/project/register", {"project_name": project})
    project_payload = response.get("project")
    if not isinstance(project_payload, dict) or not project_payload.get("id"):
        raise RuntimeError(f"Could not resolve project id for {project!r}")
    return str(project_payload["id"])


def function_definition(
    *,
    project_id: str,
    project: str,
    prompt_path: Path,
    model: str,
    if_exists: str,
    facet_key: str,
    definition: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not definition.prompt_text:
        raise ValueError(f"Facet {definition.facet_name!r} has no prompt text")

    if args.facet_name:
        name = args.facet_name
    elif len(args.facet or []) == 1:
        name = definition.facet_name
    else:
        name = f"{definition.facet_name} facet"

    slug = args.slug or (
        slugify(args.facet_name)
        if args.facet_name
        else slugify(f"{args.slug_prefix}-{facet_key}")
    )
    return {
        "project_id": project_id,
        "name": name,
        "slug": slug,
        "description": f"Facet optimizer definition for {definition.facet_name}",
        "function_type": "facet",
        "function_data": {
            "type": "facet",
            "prompt": definition.prompt_text,
            "model": model,
            "no_match_pattern": definition.no_match_pattern,
        },
        "if_exists": if_exists,
        "metadata": {
            "source": "facet-optimizer",
            "project": project,
            "prompt_file": str(prompt_path),
            "facet_name": definition.facet_name,
            "prompt_sha256": definition.prompt_sha256,
        },
        "tags": ["facet-optimizer", definition.facet_name],
    }


def main() -> int:
    args = parse_args()
    env_file = args.env_file.expanduser()
    if env_file.exists():
        load_dotenv(env_file)

    prompt_path = resolve_prompt_path(args)
    definitions = selected_definitions(args, prompt_path)
    project = resolve_project(args)
    model = args.model or os.getenv("FACET_OPTIMIZER_MODEL", "brain-facet-1")
    api_key = os.getenv("BRAINTRUST_API_KEY")
    app_url = os.getenv("BRAINTRUST_APP_URL")
    org_name = args.target_org or os.getenv("FACET_OPTIMIZER_TARGET_ORG")
    if not api_key:
        raise SystemExit("error: BRAINTRUST_API_KEY is required")

    login(api_key=api_key, app_url=app_url, org_name=org_name)
    project_id = project_id_for_name(project)
    functions = [
        function_definition(
            project_id=project_id,
            project=project,
            prompt_path=prompt_path,
            model=model,
            if_exists=args.if_exists,
            facet_key=facet_key,
            definition=definition,
            args=args,
        )
        for facet_key, definition in sorted(definitions.items())
    ]

    if args.dry_run:
        print(json.dumps({"functions": functions}, indent=2, sort_keys=True))
        return 0

    response = api_conn().post_json("insert-functions", {"functions": functions})
    print(
        json.dumps(
            {
                "project": project,
                "project_id": project_id,
                "prompt": str(prompt_path),
                "model": model,
                "published": [
                    {
                        "name": function["name"],
                        "slug": function["slug"],
                        "facet_name": function["metadata"]["facet_name"],
                    }
                    for function in functions
                ],
                "response": response,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
