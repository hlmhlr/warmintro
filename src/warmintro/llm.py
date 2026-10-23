"""Create the answer model (LLM) and the embedding model.

Replaces the lab's llm_interface.py (WatsonxLLM and WatsonxEmbeddings).
The lab's change_llm_model() changed a global setting, so in the web app one
user's choice switched the model for everyone. Here the model is an argument.

Providers: Ollama (local), Gemini, and three free OpenAI-compatible APIs
(Groq, Cloudflare Workers AI, OpenRouter). LLM_PROVIDER may list several,
e.g. "groq,cloudflare,openrouter": they are tried in that order.
"""

import logging

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM

from warmintro.config import (
    ANSWER_GENERATION,
    EMBED_DIMENSIONS,
    Generation,
    Settings,
    load_settings,
)
from warmintro.fallback import FallbackLLM

logger = logging.getLogger(__name__)

OLLAMA_CONTEXT_WINDOW = 8192      # tokens the local model can read at once
GEMINI_CONTEXT_WINDOW = 1_000_000
# Gemini's output limit includes its hidden "thinking" tokens: with only the
# answer's 250 tokens, thinking could use them all and the reply failed with
# MAX_TOKENS. Thinking is kept low and gets this much extra room; answer
# length is still set by the prompts ("one to three sentences").
GEMINI_THINKING_HEADROOM = 1024
OLLAMA_TIMEOUT = 180.0            # seconds; the first call also loads the model
# Cloud calls give up after this long, so a busy provider can't hang the app
# (Gemini once waited for minutes without answering) and the next one is tried.
CLOUD_TIMEOUT = 45.0  # below the OpenAI client default of 60, so it is really ours
# Reasoning models (gpt-oss on Groq, some OpenRouter models) count their
# reasoning tokens in the output limit, like Gemini's thinking.
REASONING_HEADROOM = 1024

# The free OpenAI-compatible providers: base URL, context window, extra
# request fields. {account} is filled in for Cloudflare.
OPENAI_LIKE = {
    "groq": ("https://api.groq.com/openai/v1", 131_072,
             # OpenAILike sends its own reasoning_effort only for OpenAI's
             # o-series models, so it goes here; hide the reasoning text.
             {"reasoning_effort": "low", "extra_body": {"include_reasoning": False}}),
    "cloudflare": ("https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
                   24_000, {}),
    "openrouter": ("https://openrouter.ai/api/v1", 32_768, {}),
}
OPENROUTER_HEADERS = {"HTTP-Referer": "https://github.com/hlmhlr/warmintro",
                      "X-Title": "warmintro"}


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

    `model` picks another model of a single provider, e.g. "granite4:micro-h".
    With a list of providers, each uses its configured model, and providers
    without an API key are skipped.
    """
    settings = settings or load_settings()
    providers = settings.llm_providers
    if len(providers) == 1:
        return _build_llm(providers[0], generation, model, settings)

    llms, names = [], []
    for provider in providers:
        try:
            llms.append(_build_llm(provider, generation, None, settings))
            names.append(provider)
        except ValueError as error:  # e.g. no API key for this provider
            logger.warning("Skipping %s: %s", provider, error)
    if not llms:
        raise ValueError(f"None of LLM_PROVIDER={settings.llm_provider} is set up. "
                         "Add their API keys to .env.")
    return FallbackLLM(llms=llms, names=names)


def _build_llm(provider: str, generation: Generation, model: str | None,
               settings: Settings) -> LLM:
    options = generation.options()

    if provider == "ollama":
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

    if provider == "gemini":
        return _build_gemini(options, model, settings)
    return _build_openai_like(provider, options, model, settings)


def _build_gemini(options: dict, model: str | None, settings: Settings) -> LLM:
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
        max_retries=1,
        http_options=types.HttpOptions(timeout=int(CLOUD_TIMEOUT * 1000)),  # milliseconds
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


def _build_openai_like(provider: str, options: dict, model: str | None,
                       settings: Settings) -> LLM:
    try:
        from llama_index.llms.openai_like import OpenAILike
    except ImportError as error:
        raise MissingExtra(f"LLM_PROVIDER={provider}", "cloud", error) from error

    key, default_model = {
        "groq": (settings.groq_api_key, settings.groq_model),
        "cloudflare": (settings.cloudflare_api_token, settings.cloudflare_model),
        "openrouter": (settings.openrouter_api_key, settings.openrouter_model),
    }[provider]
    if not key:
        env = {"groq": "GROQ_API_KEY", "cloudflare": "CLOUDFLARE_API_TOKEN",
               "openrouter": "OPENROUTER_API_KEY"}[provider]
        raise ValueError(f"LLM_PROVIDER={provider} needs {env} in .env")
    if provider == "cloudflare" and not settings.cloudflare_account_id:
        raise ValueError("LLM_PROVIDER=cloudflare needs CLOUDFLARE_ACCOUNT_ID in .env")

    base_url, context_window, extra = OPENAI_LIKE[provider]
    headroom = 0 if provider == "cloudflare" else REASONING_HEADROOM
    return OpenAILike(
        model=model or default_model,
        api_base=base_url.format(account=settings.cloudflare_account_id),
        api_key=key,
        is_chat_model=True,
        context_window=context_window,
        temperature=options["temperature"],
        max_tokens=options["max_new_tokens"] + headroom,
        # top_k isn't part of the OpenAI API; greedy decoding is temperature 0.
        additional_kwargs={"top_p": options["top_p"], **extra},
        default_headers=OPENROUTER_HEADERS if provider == "openrouter" else None,
        timeout=CLOUD_TIMEOUT,
        max_retries=1,  # the next provider is the retry
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

    if settings.embeddings_provider == "cloudflare":
        try:
            from llama_index.embeddings.openai_like import OpenAILikeEmbedding
        except ImportError as error:
            raise MissingExtra("EMBEDDINGS_PROVIDER=cloudflare", "cloud", error) from error
        if not (settings.cloudflare_api_token and settings.cloudflare_account_id):
            raise ValueError("EMBEDDINGS_PROVIDER=cloudflare needs CLOUDFLARE_API_TOKEN "
                             "and CLOUDFLARE_ACCOUNT_ID in .env")
        return OpenAILikeEmbedding(
            model_name=settings.cloudflare_embed_model,  # bge-base: 768 numbers
            api_base=OPENAI_LIKE["cloudflare"][0].format(
                account=settings.cloudflare_account_id),
            api_key=settings.cloudflare_api_token,
            timeout=CLOUD_TIMEOUT,
            max_retries=2,
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


def _model_of(provider: str, s: Settings) -> str:
    return {"ollama": s.ollama_model, "gemini": s.gemini_model, "groq": s.groq_model,
            "cloudflare": s.cloudflare_model, "openrouter": s.openrouter_model}[provider]


def describe_llm(settings: Settings | None = None) -> str:
    """The LLM(s) in use: "granite4:tiny-h (ollama)", or a chain joined by →."""
    s = settings or load_settings()
    return " → ".join(f"{_model_of(p, s)} ({p})" for p in s.llm_providers)


def describe(settings: Settings | None = None) -> str:
    """One line naming the models in use, for scripts and the web app."""
    s = settings or load_settings()
    embed = {
        "ollama": s.ollama_embed_model,
        "huggingface": s.hf_embed_model,
        "gemini": s.gemini_embed_model,
        "cloudflare": s.cloudflare_embed_model,
    }[s.embeddings_provider]
    return f"LLM: {describe_llm(s)} | embeddings: {embed} ({s.embeddings_provider})"
