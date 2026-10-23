"""All settings in one place: providers, generation presets and prompt templates.

Replaces the lab's config.py. The watsonx URL, project ID and Proxycurl key
are gone; model choices come from the .env file instead.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

LLM_PROVIDERS = ("ollama", "gemini")
EMBEDDINGS_PROVIDERS = ("ollama", "huggingface", "gemini")

# Retrieval settings (the lab's CHUNK_SIZE and SIMILARITY_TOP_K).
CHUNK_SIZE = 400       # tokens per chunk, not characters
# The lab kept SentenceSplitter's default overlap of 200 tokens, so long
# chunks repeated half of the previous one.
CHUNK_OVERLAP = 40
SIMILARITY_TOP_K = 4   # chunks retrieved per question

# Gemini embeddings are cut to this size so they match Granite's 768.
EMBED_DIMENSIONS = 768


@dataclass(frozen=True)
class Settings:
    """Model choices read from the environment (.env)."""

    llm_provider: str
    embeddings_provider: str
    ollama_base_url: str
    ollama_model: str
    ollama_embed_model: str
    google_api_key: str
    gemini_model: str
    gemini_embed_model: str
    hf_embed_model: str


def load_settings() -> Settings:
    """Read the settings, using local defaults for anything not set."""
    load_dotenv()
    settings = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
        embeddings_provider=os.getenv("EMBEDDINGS_PROVIDER", "ollama").strip().lower(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "granite4:tiny-h"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "granite-embedding:278m"),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2"),
        hf_embed_model=os.getenv(
            "HF_EMBED_MODEL", "ibm-granite/granite-embedding-278m-multilingual"
        ),
    )
    if settings.llm_provider not in LLM_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER={settings.llm_provider!r} is not supported. "
            f"Choose one of: {', '.join(LLM_PROVIDERS)}"
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
Below is information from a person's professional profile.
---------------------
{context_str}
---------------------
Using only the profile information above, write three interesting facts about
this person's career or education. After each fact, add one friendly question
that could start a conversation about it.

Rules:
- Use only facts stated in the profile. Do not guess or add details.
- Number the facts 1, 2 and 3.

Request: {query_str}
Answer:"""

USER_QUESTION_TEMPLATE = """You answer questions about a person, using their professional profile.
---------------------
{context_str}
---------------------
Answer the question using only the profile information above.
If the profile does not contain the answer, reply exactly: I don't know.
Keep the answer to one to three sentences.

Question: {query_str}
Answer:"""
