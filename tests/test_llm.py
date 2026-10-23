import sys

import pytest

from warmintro.config import Generation, load_settings
from warmintro.llm import MissingExtra, describe, get_embeddings, get_llm


def test_ollama_llm_gets_the_lab_settings(clean_env):
    llm = get_llm(Generation(123, "sample", temperature=0.4, top_k=7, top_p=0.8))
    assert llm.model == "granite4:tiny-h"
    assert llm.temperature == 0.4
    assert llm.additional_kwargs == {"num_predict": 123, "top_k": 7, "top_p": 0.8}
    # Set explicitly so LlamaIndex doesn't ask the server for it.
    assert llm.context_window == 8192


def test_model_argument_picks_another_model(clean_env):
    assert get_llm(model="granite4:micro-h").model == "granite4:micro-h"
    assert get_llm().model == "granite4:tiny-h"  # nothing global changed


def test_ollama_embeddings(clean_env):
    emb = get_embeddings()
    assert emb.model_name == "granite-embedding:278m"
    assert emb.base_url == "http://localhost:11434"


def test_describe(clean_env):
    assert describe() == ("LLM: granite4:tiny-h (ollama) | "
                          "embeddings: granite-embedding:278m (ollama)")


def test_gemini_needs_an_api_key(clean_env):
    pytest.importorskip("llama_index.llms.google_genai")
    clean_env.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        get_llm()


def test_gemini_llm_is_created_offline_with_its_token_limit(clean_env):
    pytest.importorskip("llama_index.llms.google_genai")
    clean_env.setenv("LLM_PROVIDER", "gemini")
    clean_env.setenv("GOOGLE_API_KEY", "fake-key")
    # Without context_window and max_tokens, GoogleGenAI calls Google here
    # and fails with a fake key.
    llm = get_llm(Generation(77))
    assert llm.model == "gemini-flash-latest"
    config = llm._generation_config
    # With generation_config, GoogleGenAI ignores max_tokens, so the limit
    # must be inside the config.
    # Gemini counts thinking tokens in this limit, so it gets extra room.
    assert config["max_output_tokens"] == 77 + 1024
    assert config["thinking_config"]["thinking_level"] == "LOW"
    assert (config["temperature"], config["top_k"]) == (0.0, 1)


def test_gemini_embeddings_use_768_dimensions(clean_env):
    pytest.importorskip("llama_index.embeddings.google_genai")
    clean_env.setenv("EMBEDDINGS_PROVIDER", "gemini")
    clean_env.setenv("GOOGLE_API_KEY", "fake-key")
    emb = get_embeddings()
    assert emb.model_name == "gemini-embedding-2"
    assert emb.embedding_config == {"output_dimensionality": 768}


@pytest.mark.parametrize(
    ("variable", "provider", "module", "extra"),
    [
        ("LLM_PROVIDER", "gemini", "llama_index.llms.google_genai", "gemini"),
        ("EMBEDDINGS_PROVIDER", "gemini", "llama_index.embeddings.google_genai", "gemini"),
        ("EMBEDDINGS_PROVIDER", "huggingface", "llama_index.embeddings.huggingface", "hf"),
    ],
)
def test_missing_extra_names_the_install_command(clean_env, variable, provider, module, extra):
    clean_env.setenv(variable, provider)
    clean_env.setenv("GOOGLE_API_KEY", "fake-key")
    # A None entry in sys.modules makes "import x" fail, as if x weren't installed.
    clean_env.setitem(sys.modules, module, None)
    build = get_llm if variable == "LLM_PROVIDER" else get_embeddings
    with pytest.raises(MissingExtra, match=f"uv sync --extra {extra}"):
        build()


def test_huggingface_embeddings_load_in_float32_on_cpu(clean_env):
    # Fake torch and HF modules, so the test needs neither installed.
    import types

    fake_torch = types.SimpleNamespace(float32="float32")
    captured = {}

    class FakeHuggingFaceEmbedding:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_hf = types.SimpleNamespace(HuggingFaceEmbedding=FakeHuggingFaceEmbedding)
    clean_env.setitem(sys.modules, "torch", fake_torch)
    clean_env.setitem(sys.modules, "llama_index.embeddings.huggingface", fake_hf)
    clean_env.setenv("EMBEDDINGS_PROVIDER", "huggingface")

    get_embeddings()
    assert captured["device"] == "cpu"
    assert captured["model_kwargs"] == {"dtype": "float32"}  # bfloat16 is slow on CPU
    assert captured["model_name"] == "ibm-granite/granite-embedding-278m-multilingual"


def test_settings_object_can_be_passed_in(clean_env):
    settings = load_settings()
    assert get_llm(settings=settings).model == settings.ollama_model
