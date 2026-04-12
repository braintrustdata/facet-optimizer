from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_SOURCE_MODEL = "brain-facet-latest,brain-facet-1"
DEFAULT_GROUND_TRUTH_MODEL = "gpt-5.4"
DEFAULT_TARGET_PROJECT = "Facet Optimizer"
DEFAULT_PROXY_API_BASE = "https://braintrustproxy.com/v1"
DEFAULT_CREATED_AFTER_SQL = "NOW() - INTERVAL 7 DAY"
DEFAULT_CREATED_BEFORE_SQL = "NOW() - INTERVAL 1 HOUR"
DEFAULT_CONCURRENCY = 128
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_APP_URL = "https://www.braintrust.dev"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_env_file() -> Path:
    return repo_root() / ".env"


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def env_value(name: str, default: str | None = None) -> str | None:
    return _clean(os.getenv(name)) or default


@dataclass(frozen=True)
class CreateDatasetConfig:
    env_file: Path
    source_project: str
    target_project: str
    dataset: str
    source_org: str | None
    target_org: str | None
    source_model: str
    ground_truth_model: str
    ground_truth_api_base: str | None
    ground_truth_api_key: str
    braintrust_api_key: str
    app_url: str | None
    facet_name: str | None
    positive_limit: int
    negative_limit: int
    limit: int | None
    created_after_sql: str
    created_before_sql: str
    extra_where_sql: str | None
    root_span_ids: list[str]
    concurrency: int
    validation_fraction: float
    output_root: Path
    dry_run: bool
    skip_upload: bool


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", type=Path, default=default_env_file())
    parser.add_argument("--source-project")
    parser.add_argument("--target-project")
    parser.add_argument("--dataset")
    parser.add_argument("--source-org")
    parser.add_argument("--target-org")
    parser.add_argument("--source-model")
    parser.add_argument("--ground-truth-model")
    parser.add_argument("--ground-truth-api-base")
    parser.add_argument("--ground-truth-api-key")
    parser.add_argument("--facet-name")
    parser.add_argument("--positive-limit", type=int, default=None)
    parser.add_argument("--negative-limit", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--created-after-sql", default=None)
    parser.add_argument("--created-before-sql", default=None)
    parser.add_argument("--extra-where-sql", default=None)
    parser.add_argument("--root-span-id", action="append", default=[])
    parser.add_argument("--root-span-id-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(".local/facet-optimizer"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")


def load_config(args: argparse.Namespace) -> CreateDatasetConfig:
    env_file = Path(args.env_file).expanduser().resolve()
    if env_file.exists():
        load_dotenv(env_file)

    root_ids = list(args.root_span_id or [])
    if args.root_span_id_file:
        for line in args.root_span_id_file.read_text(encoding="utf-8").splitlines():
            root_id = line.strip()
            if root_id:
                root_ids.append(root_id)
    deduped_root_ids = list(dict.fromkeys(root_ids))

    source_project = _clean(args.source_project) or env_value("FACET_OPTIMIZER_SOURCE_PROJECT")
    target_project = (
        _clean(args.target_project)
        or env_value("FACET_OPTIMIZER_TARGET_PROJECT")
        or env_value("FACET_OPTIMIZER_PROJECT")
        or DEFAULT_TARGET_PROJECT
    )
    dataset = _clean(args.dataset) or env_value("FACET_OPTIMIZER_DATASET")
    braintrust_api_key = env_value("BRAINTRUST_API_KEY")
    ground_truth_api_key = (
        _clean(args.ground_truth_api_key)
        or env_value("FACET_OPTIMIZER_GROUND_TRUTH_API_KEY")
        or env_value("FACET_OPTIMIZER_LABEL_API_KEY")
        or env_value("FACET_OPTIMIZER_JUDGE_API_KEY")
        or env_value("OPENAI_API_KEY")
        or braintrust_api_key
    )

    missing = []
    if not source_project:
        missing.append("--source-project or FACET_OPTIMIZER_SOURCE_PROJECT")
    if not target_project:
        missing.append("--target-project or FACET_OPTIMIZER_TARGET_PROJECT")
    if not dataset:
        missing.append("--dataset or FACET_OPTIMIZER_DATASET")
    if not braintrust_api_key:
        missing.append("BRAINTRUST_API_KEY")
    if not ground_truth_api_key:
        missing.append("FACET_OPTIMIZER_GROUND_TRUTH_API_KEY, OPENAI_API_KEY, or BRAINTRUST_API_KEY")
    if missing:
        raise ValueError("Missing required configuration: " + ", ".join(missing))

    positive_limit = args.positive_limit
    negative_limit = args.negative_limit
    if positive_limit is None:
        positive_limit = int(env_value("FACET_OPTIMIZER_POSITIVE_LIMIT", "100") or "100")
    if negative_limit is None:
        negative_limit = int(env_value("FACET_OPTIMIZER_NEGATIVE_LIMIT", "100") or "100")
    if positive_limit < 0 or negative_limit < 0:
        raise ValueError("--positive-limit and --negative-limit must be >= 0")
    concurrency = args.concurrency
    if concurrency is None:
        concurrency = int(
            env_value("FACET_OPTIMIZER_CONCURRENCY", str(DEFAULT_CONCURRENCY))
            or str(DEFAULT_CONCURRENCY)
        )
    if concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    validation_fraction = args.validation_fraction
    if validation_fraction is None:
        validation_fraction = float(
            env_value(
                "FACET_OPTIMIZER_VALIDATION_FRACTION",
                str(DEFAULT_VALIDATION_FRACTION),
            )
            or str(DEFAULT_VALIDATION_FRACTION)
        )
    if validation_fraction < 0 or validation_fraction >= 1:
        raise ValueError("--validation-fraction must be >= 0 and < 1")

    return CreateDatasetConfig(
        env_file=env_file,
        source_project=source_project,
        target_project=target_project,
        dataset=dataset,
        source_org=_clean(args.source_org) or env_value("FACET_OPTIMIZER_SOURCE_ORG"),
        target_org=_clean(args.target_org) or env_value("FACET_OPTIMIZER_TARGET_ORG"),
        source_model=_clean(args.source_model)
        or env_value("FACET_OPTIMIZER_SOURCE_MODEL")
        or DEFAULT_SOURCE_MODEL,
        ground_truth_model=_clean(args.ground_truth_model)
        or env_value("FACET_OPTIMIZER_GROUND_TRUTH_MODEL")
        or env_value("FACET_OPTIMIZER_JUDGE_MODEL")
        or DEFAULT_GROUND_TRUTH_MODEL,
        ground_truth_api_base=_clean(args.ground_truth_api_base)
        or env_value("FACET_OPTIMIZER_GROUND_TRUTH_API_BASE")
        or env_value("FACET_OPTIMIZER_LABEL_API_BASE")
        or env_value("FACET_OPTIMIZER_JUDGE_API_BASE")
        or env_value("BRAINTRUST_PROXY_API_BASE")
        or DEFAULT_PROXY_API_BASE,
        ground_truth_api_key=ground_truth_api_key,
        braintrust_api_key=braintrust_api_key,
        app_url=env_value("BRAINTRUST_APP_URL", DEFAULT_APP_URL),
        facet_name=_clean(args.facet_name) or env_value("FACET_OPTIMIZER_FACET_NAME"),
        positive_limit=positive_limit,
        negative_limit=negative_limit,
        limit=args.limit,
        created_after_sql=args.created_after_sql or DEFAULT_CREATED_AFTER_SQL,
        created_before_sql=args.created_before_sql or DEFAULT_CREATED_BEFORE_SQL,
        extra_where_sql=_clean(args.extra_where_sql)
        or env_value("FACET_OPTIMIZER_EXTRA_WHERE_SQL"),
        root_span_ids=deduped_root_ids,
        concurrency=concurrency,
        validation_fraction=validation_fraction,
        output_root=Path(args.output_root).expanduser(),
        dry_run=bool(args.dry_run),
        skip_upload=bool(args.skip_upload),
    )
