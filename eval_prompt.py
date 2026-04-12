#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import braintrust
from braintrust import Eval, init_dataset
from braintrust.score import Score
from dotenv import load_dotenv
from openai import OpenAI

from facet_optimizer.common import extract_output_text
from facet_optimizer.prompting import (
    is_none_like,
    load_prompt_spec,
    normalize_expected,
    render_messages,
)

load_dotenv()

PROJECT_NAME = os.getenv("FACET_OPTIMIZER_PROJECT", "Facet Optimizer")
DATASET_NAME = os.getenv("FACET_OPTIMIZER_DATASET", "")
PROMPT_PATH_RAW = os.getenv("FACET_OPTIMIZER_PROMPT_PATH", "").strip()
MODEL_NAME = os.getenv("FACET_OPTIMIZER_MODEL", "brain-facet-1")
API_BASE = os.getenv("BRAINTRUST_PROXY_API_BASE", "https://braintrustproxy.com/v1")
API_KEY = os.getenv("BRAINTRUST_API_KEY", "")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("FACET_OPTIMIZER_TIMEOUT_SECONDS", "180"))
MAX_TOKENS = int(os.getenv("FACET_OPTIMIZER_MAX_TOKENS", "20000"))
MAX_CONCURRENCY = int(os.getenv("FACET_OPTIMIZER_MAX_CONCURRENCY", "12"))

if not DATASET_NAME:
    raise ValueError("FACET_OPTIMIZER_DATASET is required")
if not PROMPT_PATH_RAW:
    raise ValueError("FACET_OPTIMIZER_PROMPT_PATH is required")
if not API_KEY:
    raise ValueError("BRAINTRUST_API_KEY is required")

PROMPT_PATH = Path(PROMPT_PATH_RAW).expanduser().resolve()
if not PROMPT_PATH.is_file():
    raise ValueError(f"FACET_OPTIMIZER_PROMPT_PATH does not exist: {PROMPT_PATH}")

PROMPT_SPEC = load_prompt_spec(PROMPT_PATH)
EXPERIMENT_NAME = os.getenv(
    "FACET_OPTIMIZER_EXPERIMENT_NAME",
    f"facet-prompt-{PROMPT_SPEC['facet_name']}-{PROMPT_SPEC['prompt_version']}",
)

CLIENT = braintrust.wrap_openai(
    OpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
)


def _task(input_data: dict[str, Any]) -> str:
    preprocessed_text = input_data["preprocessed_text"]
    messages, suffix_messages = render_messages(
        PROMPT_SPEC,
        preprocessed_text=preprocessed_text,
    )
    kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
    }
    if suffix_messages:
        kwargs["extra_body"] = {"suffix_messages": suffix_messages}

    response = CLIENT.chat.completions.create(**kwargs)
    choice = response.choices[0]
    return extract_output_text(
        [
            {
                "message": {
                    "content": choice.message.content,
                }
            }
        ]
    )


def normalized_exact_match(output: Any, expected: Any) -> Score:
    normalized_output = normalize_expected(output, prompt_spec=PROMPT_SPEC)
    normalized_expected = normalize_expected(expected, prompt_spec=PROMPT_SPEC)
    return Score(
        name="normalized_exact_match",
        score=1.0 if normalized_output == normalized_expected else 0.0,
        metadata={
            "normalized_output": normalized_output,
            "normalized_expected": normalized_expected,
        },
    )


def none_decision_match(output: Any, expected: Any) -> Score:
    output_is_none = is_none_like(output, prompt_spec=PROMPT_SPEC)
    expected_is_none = is_none_like(expected, prompt_spec=PROMPT_SPEC)
    return Score(
        name="none_decision_match",
        score=1.0 if output_is_none == expected_is_none else 0.0,
        metadata={
            "output_is_none": output_is_none,
            "expected_is_none": expected_is_none,
        },
    )


def output_nonempty(output: Any, expected: Any) -> Score:
    text = str(output or "").strip()
    return Score(
        name="output_nonempty",
        score=1.0 if bool(text) else 0.0,
    )


Eval(
    PROJECT_NAME,
    data=init_dataset(project=PROJECT_NAME, name=DATASET_NAME, use_output=False),
    task=_task,
    scores=[output_nonempty, none_decision_match, normalized_exact_match],
    experiment_name=EXPERIMENT_NAME,
    metadata={
        "prompt_path": str(PROMPT_PATH),
        "prompt_version": PROMPT_SPEC["prompt_version"],
        "facet_name": PROMPT_SPEC["facet_name"],
        "model": MODEL_NAME,
        "api_base": API_BASE,
        "max_tokens": MAX_TOKENS,
    },
    max_concurrency=MAX_CONCURRENCY,
)


if __name__ == "__main__":
    pass
