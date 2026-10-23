"""All settings in one place: providers, generation presets and prompt templates.

Replaces the lab's config.py. The watsonx URL, project ID and Proxycurl key
are gone; model choices come from the .env file instead.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

LLM_PROVIDERS = ("ollama", "gemini", "groq", "cloudflare", "openrouter")
EMBEDDINGS_PROVIDERS = ("ollama", "huggingface", "gemini", "cloudflare")
# The live demo's chain: if one free provider is busy or out of quota, the
# next one answers. Groq and Cloudflare serve a fixed model; OpenRouter's
# free router picks a random free model, so it comes last.
CLOUD_CHAIN = "groq,cloudflare,openrouter"

# Retrieval settings (the lab's CHUNK_SIZE and SIMILARITY_TOP_K).
CHUNK_SIZE = 400       # tokens per chunk, not characters
# The lab kept SentenceSplitter's default overlap of 200 tokens, so long
# chunks repeated half of the previous one.
CHUNK_OVERLAP = 40
SIMILARITY_TOP_K = 4   # chunks retrieved per question
# Facts read the whole profile instead of retrieving for a vague request: in
# the baseline scorecard, the top 8 chunks held no job in 6 of 9 runs. A
# sample profile is about 13 short chunks; very long profiles are capped.
FACTS_MAX_CHUNKS = 30
FACTS_QUERY = "Provide three interesting facts about this person's career or education."

# Gemini embeddings are cut to this size so they match Granite's 768
# (Cloudflare's bge-base model has 768 too).
EMBED_DIMENSIONS = 768


@dataclass(frozen=True)
class Settings:
    """Model choices read from the environment (.env)."""

    llm_provider: str          # one provider, or a comma list tried in order
    embeddings_provider: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "granite4:tiny-h"
    ollama_embed_model: str = "granite-embedding:278m"
    google_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    gemini_embed_model: str = "gemini-embedding-2"
    hf_embed_model: str = "ibm-granite/granite-embedding-278m-multilingual"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    cloudflare_api_token: str = ""
    cloudflare_account_id: str = ""
    cloudflare_model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
    cloudflare_embed_model: str = "@cf/baai/bge-base-en-v1.5"
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"

    @property
    def llm_providers(self) -> tuple[str, ...]:
        """LLM_PROVIDER as a tuple: "groq,cloudflare" -> ("groq", "cloudflare")."""
        return tuple(p.strip() for p in self.llm_provider.split(",") if p.strip())


_ENV_NAMES = {  # Settings field -> environment variable
    "ollama_base_url": "OLLAMA_BASE_URL", "ollama_model": "OLLAMA_MODEL",
    "ollama_embed_model": "OLLAMA_EMBED_MODEL", "google_api_key": "GOOGLE_API_KEY",
    "gemini_model": "GEMINI_MODEL", "gemini_embed_model": "GEMINI_EMBED_MODEL",
    "hf_embed_model": "HF_EMBED_MODEL", "groq_api_key": "GROQ_API_KEY",
    "groq_model": "GROQ_MODEL", "cloudflare_api_token": "CLOUDFLARE_API_TOKEN",
    "cloudflare_account_id": "CLOUDFLARE_ACCOUNT_ID", "cloudflare_model": "CLOUDFLARE_MODEL",
    "cloudflare_embed_model": "CLOUDFLARE_EMBED_MODEL",
    "openrouter_api_key": "OPENROUTER_API_KEY", "openrouter_model": "OPENROUTER_MODEL",
}


def load_settings() -> Settings:
    """Read the settings, using local defaults for anything not set."""
    load_dotenv()
    values = {field: os.getenv(env).strip() for field, env in _ENV_NAMES.items()
              if os.getenv(env, "").strip()}
    settings = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower().replace(" ", ""),
        embeddings_provider=os.getenv("EMBEDDINGS_PROVIDER", "ollama").strip().lower(),
        **values,
    )
    unknown = [p for p in settings.llm_providers if p not in LLM_PROVIDERS]
    if unknown or not settings.llm_providers:
        raise ValueError(
            f"LLM_PROVIDER={settings.llm_provider!r} is not supported. "
            f"Choose one of: {', '.join(LLM_PROVIDERS)} (or a comma list, e.g. {CLOUD_CHAIN})"
        )
    if settings.embeddings_provider not in EMBEDDINGS_PROVIDERS:
        raise ValueError(
            f"EMBEDDINGS_PROVIDER={settings.embeddings_provider!r} is not supported. "
            f"Choose one of: {', '.join(EMBEDDINGS_PROVIDERS)}"
        )
    return settings


@dataclass(frozen=True)
class Generation:
    """How the model writes, with the lab's parameter names.

    "greedy" always picks the most likely word, so it ignores temperature.
    "sample" picks randomly among likely words, so it needs temperature > 0.
    (The lab's min_new_tokens is left out: Ollama and Gemini have no such setting.)
    """

    max_new_tokens: int
    decoding_method: str = "greedy"
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 1.0

    def __post_init__(self):
        if self.decoding_method not in ("greedy", "sample"):
            raise ValueError("decoding_method must be 'greedy' or 'sample'")
        # The lab asks for sampling at temperature 0, which contradicts itself.
        if self.decoding_method == "sample" and self.temperature <= 0:
            raise ValueError("decoding_method='sample' needs a temperature above 0")

    def options(self) -> dict:
        """The settings the model actually receives."""
        if self.decoding_method == "greedy":
            return {"temperature": 0.0, "top_k": 1, "top_p": 1.0,
                    "max_new_tokens": self.max_new_tokens}
        return {"temperature": self.temperature, "top_k": self.top_k,
                "top_p": self.top_p, "max_new_tokens": self.max_new_tokens}


# Facts: a little variety reads more naturally. Answers: repeatable.
FACTS_GENERATION = Generation(max_new_tokens=500, decoding_method="sample",
                              temperature=0.3, top_k=40, top_p=0.9)
ANSWER_GENERATION = Generation(max_new_tokens=250, decoding_method="greedy")

# Prompt templates. LlamaIndex fills {context_str} with the retrieved profile
# chunks and {query_str} with the request or the user's question.
INITIAL_FACTS_TEMPLATE = """You help someone prepare for a first conversation at a networking event.
Below is a person's professional profile.
---------------------
{context_str}
---------------------
Write three interesting facts about this person's career or education, taken
only from the profile above. After each fact, write one friendly question to
ask the person about it.

Rules:
- Start your reply with "1." Write no introduction and no closing line.
- Each fact is one sentence that states only what the profile says. Copy
  names, dates and numbers exactly as written; never add new ones.
- Do not add opinions or explanations, such as why a fact is impressive.
- Take the facts from different parts of the profile, and include at least
  one job if the profile lists any.
- Write each question to the person, using "you".

Use exactly this layout:
1. <fact>
   Question: <question>
2. <fact>
   Question: <question>
3. <fact>
   Question: <question>

Request: {query_str}
Answer:"""

USER_QUESTION_TEMPLATE = """You answer questions about a person, using their professional profile.
---------------------
{context_str}
---------------------
Answer the question using only the profile information above.

Rules:
- If the profile does not contain the answer, reply exactly: I don't know.
- Keep details exactly as the profile gives them, such as "basic" or
  "intermediate" for a language, dates, and whether a job is current or past.
- If the question asks for an opinion or a guess, give only the related facts
  from the profile.
- Keep the answer to one to three sentences.

Question: {query_str}
Answer:"""
