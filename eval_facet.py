from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

from autoevals import Factuality
from braintrust import Eval, EvalHooks, init_dataset
from dotenv import load_dotenv

from facet_optimizer.eval_utils import (
    binary_classification_scores,
    sentiment_label_correct,
)
from facet_optimizer.facet_definitions import latest_prompt_path, load_facet_definitions
from facet_optimizer.facet_runtime import FacetModel


load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

EVAL_PROJECT_NAME = os.getenv("FACET_OPTIMIZER_EVAL_PROJECT") or os.getenv(
    "FACET_OPTIMIZER_TARGET_PROJECT", "Facet Optimizer"
)
DATASET_PROJECT_NAME = os.getenv(
    "FACET_OPTIMIZER_DATASET_PROJECT", EVAL_PROJECT_NAME
)
DATASET_NAME = os.getenv("FACET_OPTIMIZER_DATASET", "Facet groundtruth")
EXPERIMENT_PREFIX = os.getenv("FACET_OPTIMIZER_EXPERIMENT_PREFIX", "facet-optimizer")
MODEL = os.getenv("FACET_OPTIMIZER_MODEL", "brain-facet-1")
API_BASE = os.getenv("FACET_OPTIMIZER_API_BASE") or os.getenv(
    "BRAINTRUST_PROXY_API_BASE", "https://braintrustproxy.com/v1"
)
API_KEY = os.getenv("FACET_OPTIMIZER_API_KEY") or os.getenv("BRAINTRUST_API_KEY")
PROMPT_PATH = os.getenv("FACET_OPTIMIZER_PROMPT")
OUTPUT_ROOT = os.getenv("FACET_OPTIMIZER_OUTPUT_ROOT", ".local/facet-optimizer")
MAX_TOKENS = int(os.getenv("FACET_OPTIMIZER_MAX_TOKENS", "20000"))
REQUEST_TIMEOUT = float(os.getenv("FACET_OPTIMIZER_REQUEST_TIMEOUT", "120"))
MAX_CONCURRENCY = int(os.getenv("FACET_OPTIMIZER_MAX_CONCURRENCY", "16"))
TRIAL_COUNT = int(os.getenv("FACET_OPTIMIZER_TRIAL_COUNT", "1"))
MAX_ROWS = int(os.getenv("FACET_OPTIMIZER_MAX_ROWS", "0") or "0")
FACET_FILTER = {
    item.strip().lower()
    for item in os.getenv("FACET_OPTIMIZER_FACET_FILTER", "").split(",")
    if item.strip()
}
SCOPED_DATASET_IDS = {
    item.strip()
    for item in os.getenv("FACET_OPTIMIZER_DATASET_IDS", "").split(",")
    if item.strip()
}
SPLIT_FILTER = {
    item.strip().lower()
    for item in os.getenv("FACET_OPTIMIZER_SPLIT", "").split(",")
    if item.strip()
}
FACTUALITY_MODEL = "gpt-5.4"


def _resolve_prompt_path() -> Path:
    if PROMPT_PATH:
        return Path(PROMPT_PATH).expanduser()
    latest = latest_prompt_path(OUTPUT_ROOT)
    if latest is None:
        raise ValueError(
            "Set FACET_OPTIMIZER_PROMPT or run create_facet_dataset.py first"
        )
    return latest


PROMPT_FILE = _resolve_prompt_path()
FACETS = load_facet_definitions(PROMPT_FILE)
MODEL_CLIENT = FacetModel(
    model=MODEL,
    api_key=API_KEY or "",
    api_base=API_BASE,
    max_tokens=MAX_TOKENS,
    request_timeout=REQUEST_TIMEOUT,
)


def _row_facet(input_value: Any) -> str:
    if not isinstance(input_value, dict):
        return ""
    return str(input_value.get("facet_name") or "").strip().lower()


def data_generator():
    dataset = init_dataset(project=DATASET_PROJECT_NAME, name=DATASET_NAME)
    yielded = 0
    for row in dataset:
        row_id = str(row.get("id") or "").strip()
        input_value = row.get("input")
        facet_name = _row_facet(input_value)
        row_metadata = dict(row.get("metadata") or {})
        row_split = str(row_metadata.get("split") or "").strip().lower()
        if SCOPED_DATASET_IDS and row_id not in SCOPED_DATASET_IDS:
            continue
        if FACET_FILTER and facet_name not in FACET_FILTER:
            continue
        if SPLIT_FILTER and row_split not in SPLIT_FILTER:
            continue

        row_metadata["dataset_row_id"] = row_id
        row_metadata["prompt_file"] = str(PROMPT_FILE)
        yielded_row = dict(row)
        yielded_row["metadata"] = row_metadata
        yield yielded_row

        yielded += 1
        if MAX_ROWS and yielded >= MAX_ROWS:
            break


async def task(input: dict[str, Any], hooks: EvalHooks) -> str:
    facet_name = _row_facet(input)
    definition = FACETS.get(facet_name)
    if definition is None:
        raise ValueError(
            f"Unknown facet {facet_name!r}; known facets: {', '.join(sorted(FACETS))}"
        )
    preprocessed_text = str(input.get("preprocessed_text") or "")

    hooks.metadata["facet_name"] = definition.facet_name
    hooks.metadata["facet_prompt_sha256"] = definition.prompt_sha256
    hooks.metadata["facet_model"] = MODEL
    hooks.metadata["facet_prompt_file"] = str(PROMPT_FILE)

    return await MODEL_CLIENT.run(
        definition=definition,
        preprocessed_text=preprocessed_text,
    )


def scores():
    return [
        binary_classification_scores,
        sentiment_label_correct,
        Factuality.partial(model=FACTUALITY_MODEL),
    ]


Eval(
    EVAL_PROJECT_NAME,
    data=data_generator,
    task=task,
    scores=scores(),
    experiment_name="-".join([EXPERIMENT_PREFIX, MODEL]),
    metadata={
        "dataset_project": DATASET_PROJECT_NAME,
        "dataset": DATASET_NAME,
        "model": MODEL,
        "prompt_file": str(PROMPT_FILE),
        "max_tokens": MAX_TOKENS,
        "request_timeout": REQUEST_TIMEOUT,
        "trial_count": TRIAL_COUNT,
        "max_rows": MAX_ROWS or None,
        "facet_filter": sorted(FACET_FILTER) if FACET_FILTER else None,
        "split_filter": sorted(SPLIT_FILTER) if SPLIT_FILTER else None,
        "scoped_dataset_ids_count": len(SCOPED_DATASET_IDS),
        "factuality_model": FACTUALITY_MODEL,
        "prompt_hashes": {
            name: definition.prompt_sha256 for name, definition in sorted(FACETS.items())
        },
    },
    trial_count=TRIAL_COUNT,
    max_concurrency=MAX_CONCURRENCY,
)
