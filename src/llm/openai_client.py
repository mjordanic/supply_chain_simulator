"""Thin OpenAI structured-output client.

Replaces the deleted ``src/utils/ai_client.py`` (Anthropic+OpenAI shim).
The ``LLMClient`` ``Protocol`` is the test injection seam consumed by
``WorldBuilder``; ``OpenAIClient`` is the production implementation that
wraps ``openai.OpenAI`` and calls the structured-output ``chat.completions.parse``
endpoint so OpenAI validates the response against the Pydantic schema
server-side.

Tests inject a ``CannedClient`` (see ``scenarios/example_llm_world_offline.py``)
that satisfies the same Protocol without any network access.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, TypeVar, runtime_checkable

from openai import APIConnectionError
from pydantic import BaseModel


# Bound the schema TypeVar to ``BaseModel`` so ``structured_completion``
# can be statically typed as "returns an instance of the same schema".
T = TypeVar("T", bound=BaseModel)
# Module-scoped logger — set by ``logging.basicConfig`` in the
# scenarios at import time.
logger = logging.getLogger(__name__)


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
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,
    ) -> None:
        # Imported lazily so that importing ``src.llm`` doesn't require
        # ``openai`` to be installed when only the schemas are needed.
        from openai import OpenAI

        # Load a repo-root ``.env`` into ``os.environ`` so the env-var
        # fallback below picks up a key kept on disk. ``os.getenv`` only
        # sees vars already in the environment; ``.env`` is inert until
        # parsed. No-op if already loaded or absent; never overrides a
        # var already set in the real environment.
        from dotenv import load_dotenv

        load_dotenv()

        # Prefer the kwarg; fall back to the conventional env var.
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAIClient: OPENAI_API_KEY env var or api_key kwarg required"
            )
        # Underlying SDK client; one per ``OpenAIClient`` instance.
        self.client = OpenAI(api_key=key)
        # Default model identifier — overridable per-instance.
        self.model = model
        # Capture whether a custom base URL is set so the diagnostic
        # log line includes it (helps when a localhost proxy is
        # silently intercepting calls).
        base_via_env = (
            os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_URL") or ""
        ).strip()
        logger.info(
            "OpenAI client ready model=%s base_url=%s (OPENAI_BASE_URL in env=%s)",
            self.model,
            str(self.client.base_url).rstrip("/"),
            "yes" if base_via_env else "no",
        )

    def structured_completion(
        self, *, system: str, user: str, schema: type[T]
    ) -> T:
        """Issue one structured-output call; return the parsed schema instance.

        Errors raised:

        - ``pydantic.ValidationError`` (re-raised from SDK parsing) —
          caller may retry with the error fed back into the prompt.
        - ``openai.APIConnectionError`` — network/proxy failure; not
          retried here.
        - ``RuntimeError`` for a refusal or a missing parsed payload.
        """
        logger.debug(
            "OpenAI completions.parse start schema=%s system_chars=%s user_chars=%s",
            schema.__name__,
            len(system),
            len(user),
        )
        try:
            # ``parse`` returns a parsed response with ``message.parsed``
            # already bound to an instance of ``schema``.
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=schema,
            )
        except APIConnectionError as e:
            # Surface enough context to debug the most common cause —
            # a stale ``OPENAI_BASE_URL`` env var pointing at localhost.
            base = str(self.client.base_url).rstrip("/")
            cause = e.__cause__
            logger.error(
                "OpenAI connection failure model=%s base_url=%s schema=%s: %s "
                "(underlying: %s). If you see connection refused to localhost, "
                "unset or fix OPENAI_BASE_URL / proxy env (HTTP_PROXY, HTTPS_PROXY).",
                self.model,
                base,
                schema.__name__,
                e,
                repr(cause) if cause else "none",
            )
            raise
        # Top-level message from the first (and only) choice.
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            # The model declined to answer — bubble up so the caller can
            # decide whether to retry with a different prompt.
            raise RuntimeError(f"OpenAI refused completion: {message.refusal}")
        if message.parsed is None:
            # Should not happen with ``response_format=schema``, but
            # guard anyway and include the raw content for debugging.
            raise RuntimeError(
                "OpenAI returned no parsed payload for "
                f"{schema.__name__}; raw content: {message.content!r}"
            )
        return message.parsed


__all__ = ["LLMClient", "OpenAIClient"]
