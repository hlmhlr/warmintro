"""Create the answer model (LLM) and the embedding model.

Replaces the lab's llm_interface.py (WatsonxLLM and WatsonxEmbeddings).
The lab's change_llm_model() changed a global setting, so in the web app one
user's choice switched the model for everyone. Here the model is an argument.
"""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM

from warmintro.config import (
    ANSWER_GENERATION,
    EMBED_DIMENSIONS,
    Generation,
    Settings,
    load_settings,
)

OLLAMA_CONTEXT_WINDOW = 8192      # tokens the local model can read at once
GEMINI_CONTEXT_WINDOW = 1_000_000
# Gemini's output limit includes its hidden "thinking" tokens: with only the
# answer's 250 tokens, thinking could use them all and the reply failed with
# MAX_TOKENS. Thinking is kept low and gets this much extra room; answer
# length is still set by the prompts ("one to three sentences").
GEMINI_THINKING_HEADROOM = 1024
OLLAMA_TIMEOUT = 180.0            # seconds; the first call also loads the model


class MissingExtra(RuntimeError):
    """An optional package is needed for the chosen provider."""

    def __init__(self, setting: str, extra: str, error: ImportError):
        super().__init__(
            f"{setting} needs the '{extra}' packages ({error.name} is missing). "
            f"Install them with: uv sync --extra {extra}"
        )


def get_llm(
    generation: Generation = ANSWER_GENERATION,
    model: str | None = None,
    settings: Settings | None = None,
) -> LLM:
    """Return the chat model from LLM_PROVIDER, with these generation settings.

    `model` picks another model of the same provider, e.g. "granite4:micro-h".
    """
    settings = settings or load_settings()
    options = generation.options()

    if settings.llm_provider == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=model or settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=options["temperature"],
            # Set explicitly, otherwise LlamaIndex asks the server for it.
            context_window=OLLAMA_CONTEXT_WINDOW,
            request_timeout=OLLAMA_TIMEOUT,
            additional_kwargs={
                "num_predict": options["max_new_tokens"],  # Ollama's max_new_tokens
                "top_k": options["top_k"],
                "top_p": options["top_p"],
            },
        )

    # gemini
    try:
        from google.genai import types
        from llama_index.llms.google_genai import GoogleGenAI
    except ImportError as error:
        raise MissingExtra("LLM_PROVIDER=gemini", "gemini", error) from error
    if not settings.google_api_key:
        raise ValueError("LLM_PROVIDER=gemini needs GOOGLE_API_KEY in .env")

    max_tokens = options["max_new_tokens"] + GEMINI_THINKING_HEADROOM
    return GoogleGenAI(
        model=model or settings.gemini_model,
        api_key=settings.google_api_key,
        temperature=options["temperature"],
        # Both set explicitly, otherwise GoogleGenAI calls Google on creation.
        max_tokens=max_tokens,
        context_window=GEMINI_CONTEXT_WINDOW,
        # When generation_config is given, GoogleGenAI ignores max_tokens,
        # so the limit must be repeated here.
        generation_config=types.GenerateContentConfig(
            temperature=options["temperature"],
            top_k=options["top_k"],
            top_p=options["top_p"],
            max_output_tokens=max_tokens,
            # "low" is accepted by every current Flash model ("minimal" isn't).
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )


def get_embeddings(settings: Settings | None = None) -> BaseEmbedding:
    """Return the embedding model from EMBEDDINGS_PROVIDER.

    Unlike the lab, chunks are not truncated: the lab's
    truncate_input_tokens=3 embedded only the first 3 tokens of each chunk.
    """
    settings = settings or load_settings()

    if settings.embeddings_provider == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(
            model_name=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    if settings.embeddings_provider == "huggingface":
        try:
            import torch
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError as error:
            raise MissingExtra("EMBEDDINGS_PROVIDER=huggingface", "hf", error) from error

        return HuggingFaceEmbedding(
            model_name=settings.hf_embed_model,
            device="cpu",
            # The model is stored in bfloat16, which most CPUs run slowly.
            model_kwargs={"dtype": torch.float32},
        )

    # gemini
    try:
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    except ImportError as error:
        raise MissingExtra("EMBEDDINGS_PROVIDER=gemini", "gemini", error) from error
    if not settings.google_api_key:
        raise ValueError("EMBEDDINGS_PROVIDER=gemini needs GOOGLE_API_KEY in .env")

    return GoogleGenAIEmbedding(
        model_name=settings.gemini_embed_model,
        api_key=settings.google_api_key,
        embedding_config={"output_dimensionality": EMBED_DIMENSIONS},
    )


def describe(settings: Settings | None = None) -> str:
    """One line naming the models in use, for scripts and the web app."""
    s = settings or load_settings()
    llm = {"ollama": s.ollama_model, "gemini": s.gemini_model}[s.llm_provider]
    embed = {
        "ollama": s.ollama_embed_model,
        "huggingface": s.hf_embed_model,
        "gemini": s.gemini_embed_model,
    }[s.embeddings_provider]
    return (f"LLM: {llm} ({s.llm_provider}) | "
            f"embeddings: {embed} ({s.embeddings_provider})")
