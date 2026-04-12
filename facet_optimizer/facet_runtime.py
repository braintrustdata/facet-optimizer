from __future__ import annotations

import openai
from braintrust import wrap_openai

from .eval_utils import strip_thinking
from .facet_definitions import FacetDefinition, render_messages


class FacetModel:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        api_base: str | None,
        max_tokens: int,
        request_timeout: float,
    ) -> None:
        kwargs = {"api_key": api_key, "timeout": request_timeout}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = wrap_openai(openai.AsyncOpenAI(**kwargs))
        self.model = model
        self.max_tokens = max_tokens

    async def run(
        self,
        *,
        definition: FacetDefinition,
        preprocessed_text: str,
    ) -> str:
        request = {
            "model": self.model,
            "messages": render_messages(
                definition,
                preprocessed_text=preprocessed_text,
            ),
            "max_tokens": self.max_tokens,
        }
        if definition.suffix_messages:
            request["extra_body"] = {"suffix_messages": definition.suffix_messages}
        response = await self.client.chat.completions.create(**request)
        content = response.choices[0].message.content if response.choices else None
        return strip_thinking(content or "")
