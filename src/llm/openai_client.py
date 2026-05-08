"""Thin OpenAI structured-output client (issue 08).

Replaces the deleted ``src/utils/ai_client.py`` (Anthropic+OpenAI shim).
The ``LLMClient`` ``Protocol`` is the test injection seam consumed by
``WorldBuilder``; ``OpenAIClient`` is the production implementation that
wraps ``openai.OpenAI``.
"""

from __future__ import annotations

import os
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """One-method protocol: structured-output chat completion.

    Implementations parse the model's response into ``schema`` and raise
    ``pydantic.ValidationError`` on schema mismatch so the caller can feed
    the failure back into the next prompt. Refusals / transport errors
    propagate unchanged.
    """

    def structured_completion(
        self, *, system: str, user: str, schema: type[T]
    ) -> T: ...


class OpenAIClient:
    """Wrap ``openai.OpenAI`` with one structured-output completion call.

    Uses ``client.beta.chat.completions.parse`` so the SDK enforces the
    Pydantic schema server-side and surfaces a parsed instance directly.
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAIClient: OPENAI_API_KEY env var or api_key kwarg required"
            )
        self.client = OpenAI(api_key=key)
        self.model = model

    def structured_completion(
        self, *, system: str, user: str, schema: type[T]
    ) -> T:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise RuntimeError(f"OpenAI refused completion: {message.refusal}")
        if message.parsed is None:
            raise RuntimeError(
                "OpenAI returned no parsed payload for "
                f"{schema.__name__}; raw content: {message.content!r}"
            )
        return message.parsed


__all__ = ["LLMClient", "OpenAIClient"]
