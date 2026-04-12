from __future__ import annotations

import re
from typing import Any

from braintrust.score import Score


WHITESPACE_RE = re.compile(r"\s+")
THINKING_TAGS_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
NONE_PREFIX_RE = re.compile(r"^\s*(?:none|no[_ ]match|skipped)\b", re.IGNORECASE)
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive", "Mixed"]
SENTIMENT_LABEL_RE = re.compile(
    r"^\s*("
    + "|".join(
        re.escape(label) for label in sorted(SENTIMENT_LABELS, key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)
SENTIMENT_LABEL_MAP = {label.lower(): label for label in SENTIMENT_LABELS}
LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*\u2022]\s+|\d+\s*[.)]\s+|\(\d+\)\s+)")
LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:label|classification|class|sentiment|category)\s*[:\-]\s*",
    re.IGNORECASE,
)
LEADING_QUOTES_RE = re.compile(r"^\s*[\"'`]+\s*")


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def strip_thinking(text: str) -> str:
    return THINKING_TAGS_RE.sub("", text).strip()


def normalize_text(value: Any) -> str:
    text = strip_thinking(as_text(value)).lower()
    text = text.replace("’", "'").replace("-", " ").replace("_", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def is_none_like(value: Any) -> bool:
    normalized = normalize_text(value)
    if normalized in {"", "none", "no match", "skipped", "null", "na", "n/a"}:
        return True
    return bool(NONE_PREFIX_RE.match(as_text(value)))


def facet_type(input_value: Any) -> str:
    if not isinstance(input_value, dict):
        return ""
    return as_text(input_value.get("facet_name") or input_value.get("facet_type")).lower()


def normalize_label_prefix_noise(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    for _ in range(6):
        previous = cleaned
        cleaned = LEADING_LIST_MARKER_RE.sub("", cleaned)
        cleaned = LABEL_PREFIX_RE.sub("", cleaned)
        cleaned = LEADING_QUOTES_RE.sub("", cleaned)
        cleaned = cleaned.lstrip()
        if cleaned == previous:
            break
    return cleaned.strip()


def split_label_and_explanation(
    text: str, label_re: re.Pattern[str], label_map: dict[str, str]
) -> tuple[str | None, str]:
    if not text:
        return None, ""

    normalized = normalize_label_prefix_noise(text)
    if not normalized:
        return None, ""

    match = label_re.match(normalized)
    if not match:
        return None, normalized

    raw_label = match.group(1).lower()
    label = label_map.get(raw_label)
    remainder = normalized[match.end() :].lstrip(" \t\r\n-:.,)")
    return label, remainder.strip()


def split_sentiment_label_and_explanation(text: Any) -> tuple[str | None, str]:
    return split_label_and_explanation(
        as_text(text), SENTIMENT_LABEL_RE, SENTIMENT_LABEL_MAP
    )


def binary_classification_scores(input: Any, output: Any, expected: Any) -> list[Score]:
    row_facet_type = facet_type(input)
    output_positive = not is_none_like(output)
    expected_positive = not is_none_like(expected)
    output_negative = not output_positive
    expected_negative = not expected_positive

    return [
        Score(
            name="binary_decision_match",
            score=1.0 if output_positive == expected_positive else 0.0,
            metadata={
                "facet_type": row_facet_type,
                "output_is_positive": output_positive,
                "expected_is_positive": expected_positive,
                "output_is_negative": output_negative,
                "expected_is_negative": expected_negative,
            },
        ),
        Score(
            name="positive_recall",
            score=(1.0 if output_positive else 0.0) if expected_positive else None,
            metadata={
                "facet_type": row_facet_type,
                "output_is_positive": output_positive,
                "expected_is_positive": expected_positive,
            },
        ),
        Score(
            name="negative_specificity",
            score=(1.0 if output_negative else 0.0) if expected_negative else None,
            metadata={
                "facet_type": row_facet_type,
                "output_is_negative": output_negative,
                "expected_is_negative": expected_negative,
            },
        ),
    ]


def sentiment_label_correct(input: Any, output: Any, expected: Any) -> Score:
    row_facet_type = facet_type(input)
    expected_label, expected_explanation = split_sentiment_label_and_explanation(
        expected
    )
    output_label, output_explanation = split_sentiment_label_and_explanation(output)
    score = None
    if row_facet_type == "sentiment" and expected_label is not None:
        score = 1.0 if output_label == expected_label else 0.0

    return Score(
        name="sentiment_label_correct",
        score=score,
        metadata={
            "facet_type": row_facet_type,
            "expected_label": expected_label,
            "output_label": output_label,
            "expected_explanation_present": bool(expected_explanation),
            "output_explanation_present": bool(output_explanation),
        },
    )
