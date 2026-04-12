from __future__ import annotations

from datetime import UTC, datetime

from openai import OpenAI

from .models import GroundTruthResult, Message, ParsedFacetCall


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def build_ground_truth_messages(call: ParsedFacetCall) -> list[Message]:
    if call.base_messages:
        messages = [dict(message) for message in call.base_messages]
    else:
        messages = [
            {
                "role": "user",
                "content": "Here is the data to analyze:\n\n" + call.preprocessed_text,
            }
        ]
    messages.append({"role": "user", "content": call.facet_prompt})
    return messages


class GroundTruthLabeler:
    def __init__(self, *, model: str, api_key: str, api_base: str | None = None):
        kwargs = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)
        self.model = model

    def label(self, call: ParsedFacetCall) -> GroundTruthResult:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=build_ground_truth_messages(call),
            max_completion_tokens=512,
        )
        text = response.choices[0].message.content if response.choices else None
        expected = (text or "").strip()
        if not expected:
            raise ValueError("Ground-truth model returned an empty expected value")
        return GroundTruthResult(
            expected=expected,
            model=self.model,
            generated_at=utc_now(),
            raw_output=text,
        )
