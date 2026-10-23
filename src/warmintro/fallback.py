"""Try several LLM providers in order: if one fails, the next one answers.

Free API tiers fail in ordinary ways: rate limits (429), overloaded servers
(503, 504) and slow replies. On the live demo a single provider failing would
leave the user with an error, so FallbackLLM wraps a list of LLMs and moves
on to the next one when a call raises or comes back empty. If every provider
fails, it raises AllProvidersFailed with each provider's error.
"""

import logging
from typing import Any

from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)


class AllProvidersFailed(RuntimeError):
    """Every provider in the chain failed. `errors` holds (name, error) pairs."""

    def __init__(self, errors: list[tuple[str, Exception]]):
        self.errors = errors
        super().__init__("; ".join(f"{name}: {type(error).__name__}: {str(error)[:150]}"
                                   for name, error in errors))


class FallbackLLM(CustomLLM):
    """An LLM made of several LLMs, used in order until one answers."""

    llms: list[Any]
    names: list[str]
    _last_provider: str | None = PrivateAttr(default=None)

    @property
    def last_provider(self) -> str | None:
        """Name of the provider that gave the last answer."""
        return self._last_provider

    @property
    def metadata(self) -> LLMMetadata:
        # The prompt must fit the smallest context window in the chain.
        return LLMMetadata(
            context_window=min(llm.metadata.context_window for llm in self.llms),
            num_output=self.llms[0].metadata.num_output,
            model_name=" → ".join(self.names),
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        errors = []
        for name, llm in zip(self.names, self.llms):
            try:
                response = llm.complete(prompt, formatted=formatted, **kwargs)
            except Exception as error:  # any failure: try the next provider
                logger.warning("%s failed (%s: %s); trying the next provider",
                               name, type(error).__name__, str(error)[:200])
                errors.append((name, error))
                continue
            if not (response.text or "").strip():
                # E.g. a reasoning model that spent all its tokens thinking.
                logger.warning("%s returned an empty reply; trying the next provider", name)
                errors.append((name, RuntimeError("empty reply")))
                continue
            self._last_provider = name
            return response
        raise AllProvidersFailed(errors)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        yield self.complete(prompt, formatted=formatted, **kwargs)
