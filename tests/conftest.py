import pytest

import warmintro.config as config

ENV_VARS = [
    "LLM_PROVIDER", "EMBEDDINGS_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
    "OLLAMA_EMBED_MODEL", "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_EMBED_MODEL",
    "HF_EMBED_MODEL",
]


@pytest.fixture
def clean_env(monkeypatch):
    """Tests see only the variables they set, never a developer's .env file."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    return monkeypatch
