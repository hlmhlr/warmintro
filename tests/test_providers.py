"""The free cloud providers and the fallback chain, checked at the HTTP level.

Requests go to a fake transport, so the tests see exactly what each provider
would receive, without network or keys.
"""

import json

import httpx
import pytest

from fakes import ScriptedLLM
from warmintro.config import ANSWER_GENERATION, Generation, load_settings
from warmintro.errors import friendly_error
from warmintro.fallback import AllProvidersFailed, FallbackLLM
from warmintro.llm import describe, describe_llm, get_embeddings, get_llm

pytest.importorskip("llama_index.llms.openai_like")

KEYS = {"GROQ_API_KEY": "gsk-test", "CLOUDFLARE_API_TOKEN": "cf-test",
        "CLOUDFLARE_ACCOUNT_ID": "acc123", "OPENROUTER_API_KEY": "or-test"}


@pytest.fixture
def cloud_env(clean_env):
    for name in list(KEYS) + ["GROQ_MODEL", "CLOUDFLARE_MODEL", "OPENROUTER_MODEL",
                              "CLOUDFLARE_QUERY_INSTRUCTION"]:
        clean_env.delenv(name, raising=False)
    for name, value in KEYS.items():
        clean_env.setenv(name, value)
    return clean_env


def capture(llm, reply="Hello."):
    """Send the LLM's requests to a fake server; return the recorded requests."""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={
            "id": "1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply}}]})

    llm._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return requests


@pytest.mark.parametrize("provider, url, model", [
    ("groq", "https://api.groq.com/openai/v1/chat/completions", "openai/gpt-oss-120b"),
    ("cloudflare", "https://api.cloudflare.com/client/v4/accounts/acc123/ai/v1/chat/completions",
     "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions", "openrouter/free"),
])
def test_each_provider_gets_the_right_url_model_and_key(cloud_env, provider, url, model):
    cloud_env.setenv("LLM_PROVIDER", provider)
    llm = get_llm(ANSWER_GENERATION)
    requests = capture(llm)
    assert llm.complete("Hi").text == "Hello."
    [request] = requests
    body = json.loads(request.content)
    assert str(request.url) == url and body["model"] == model
    assert request.headers["authorization"] == f"Bearer {KEYS[ {'groq': 'GROQ_API_KEY', 'cloudflare': 'CLOUDFLARE_API_TOKEN', 'openrouter': 'OPENROUTER_API_KEY'}[provider]]}"
    assert body["temperature"] == 0.0 and "top_k" not in body  # top_k isn't OpenAI API


def test_groq_asks_for_low_reasoning_and_leaves_room_for_it(cloud_env):
    # OpenAILike's own reasoning_effort is sent only for OpenAI o-series models.
    cloud_env.setenv("LLM_PROVIDER", "groq")
    llm = get_llm(Generation(max_new_tokens=250))
    requests = capture(llm)
    llm.complete("Hi")
    body = json.loads(requests[0].content)
    assert body["reasoning_effort"] == "low"
    assert body["include_reasoning"] is False
    assert body["max_tokens"] == 250 + 1024


def test_cloudflare_has_no_reasoning_headroom(cloud_env):
    cloud_env.setenv("LLM_PROVIDER", "cloudflare")
    llm = get_llm(Generation(max_new_tokens=250))
    requests = capture(llm)
    llm.complete("Hi")
    assert json.loads(requests[0].content)["max_tokens"] == 250


def test_openrouter_identifies_the_app(cloud_env):
    cloud_env.setenv("LLM_PROVIDER", "openrouter")
    llm = get_llm()
    requests = capture(llm)
    llm.complete("Hi")
    assert requests[0].headers["x-title"] == "warmintro"


def test_cloud_calls_time_out(cloud_env):
    cloud_env.setenv("LLM_PROVIDER", "groq")
    assert get_llm().timeout == 45.0  # a busy provider can't hang the app


@pytest.mark.parametrize("missing, message", [
    ("GROQ_API_KEY", "needs GROQ_API_KEY"),
    ("CLOUDFLARE_ACCOUNT_ID", "needs CLOUDFLARE_ACCOUNT_ID"),
])
def test_missing_key_names_the_variable(cloud_env, missing, message):
    cloud_env.setenv("LLM_PROVIDER", "groq" if missing.startswith("GROQ") else "cloudflare")
    cloud_env.delenv(missing)
    with pytest.raises(ValueError, match=message):
        get_llm()


def test_chain_builds_a_fallback_and_skips_providers_without_keys(cloud_env):
    cloud_env.setenv("LLM_PROVIDER", "groq, cloudflare, openrouter")
    cloud_env.delenv("CLOUDFLARE_API_TOKEN")
    llm = get_llm()
    assert isinstance(llm, FallbackLLM) and llm.names == ["groq", "openrouter"]


def test_chain_without_any_key_explains_what_to_add(cloud_env):
    cloud_env.setenv("LLM_PROVIDER", "groq,openrouter")
    cloud_env.delenv("GROQ_API_KEY")
    cloud_env.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ValueError, match="Add their API keys"):
        get_llm()


def test_describe_shows_the_chain(cloud_env):
    cloud_env.setenv("LLM_PROVIDER", "groq,cloudflare")
    cloud_env.setenv("EMBEDDINGS_PROVIDER", "cloudflare")
    assert describe_llm() == ("openai/gpt-oss-120b (groq) → "
                              "@cf/meta/llama-3.3-70b-instruct-fp8-fast (cloudflare)")
    assert describe().endswith("embeddings: @cf/baai/bge-base-en-v1.5 (cloudflare)")


def test_unknown_provider_in_a_chain_is_rejected(cloud_env):
    cloud_env.setenv("LLM_PROVIDER", "groq,openai")
    with pytest.raises(ValueError, match="comma list"):
        load_settings()


def capture_embeddings(embedding):
    """Fake embedding server: records each request's input texts."""
    inputs = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        inputs.append(texts)
        return httpx.Response(200, json={"object": "list", "model": "m", "data": [
            {"object": "embedding", "index": i, "embedding": [0.1] * 768}
            for i in range(len(texts))]})

    embedding._client = None
    embedding._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return inputs


def test_cloudflare_embeddings(cloud_env):
    pytest.importorskip("llama_index.embeddings.openai_like")
    cloud_env.setenv("EMBEDDINGS_PROVIDER", "cloudflare")
    emb = get_embeddings()
    assert emb.model_name == "@cf/baai/bge-base-en-v1.5"
    assert emb.inner.api_base == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/v1"


def test_bge_instruction_goes_on_queries_only(cloud_env):
    pytest.importorskip("llama_index.embeddings.openai_like")
    cloud_env.setenv("EMBEDDINGS_PROVIDER", "cloudflare")
    emb = get_embeddings()
    inputs = capture_embeddings(emb.inner)
    emb.get_query_embedding("What is her current job?")
    emb.get_text_embedding_batch(["Current job: Amara works as Lead."])
    assert inputs == [
        ["Represent this sentence for searching relevant passages: What is her current job?"],
        ["Current job: Amara works as Lead."]]  # the model card: never on passages


def test_bge_instruction_can_be_turned_off(cloud_env):
    pytest.importorskip("llama_index.embeddings.openai_like")
    cloud_env.setenv("EMBEDDINGS_PROVIDER", "cloudflare")
    cloud_env.setenv("CLOUDFLARE_QUERY_INSTRUCTION", "none")
    emb = get_embeddings()
    inputs = capture_embeddings(emb)
    emb.get_query_embedding("What is her current job?")
    assert inputs == [["What is her current job?"]]


# ---- The fallback itself, with scripted models ------------------------

def chain(*answers):
    llms = [ScriptedLLM(answers=[a], prompts=[]) for a in answers]
    return FallbackLLM(llms=llms, names=[f"p{n}" for n in range(1, len(llms) + 1)])


def test_first_working_provider_answers():
    llm = chain("First.", "Second.")
    assert llm.complete("Hi").text == "First." and llm.last_provider == "p1"
    assert llm.llms[1].prompts == []  # the second was never called


@pytest.mark.parametrize("failure", [
    ConnectionError("down"), TimeoutError("slow"), RuntimeError("503 UNAVAILABLE"), "  "])
def test_failure_or_empty_reply_moves_to_the_next(failure):
    llm = chain(failure, "Backup answer.")
    assert llm.complete("Hi").text == "Backup answer." and llm.last_provider == "p2"


def test_all_failing_raises_with_every_error():
    llm = chain(RuntimeError("429 quota"), ConnectionError("down"))
    with pytest.raises(AllProvidersFailed) as failed:
        llm.complete("Hi")
    assert [name for name, _ in failed.value.errors] == ["p1", "p2"]
    assert "429 quota" in str(failed.value)


def test_prompt_must_fit_the_smallest_context_window():
    small = ScriptedLLM(prompts=[])
    assert chain("a", "b").metadata.context_window == small.metadata.context_window
    assert chain("a").metadata.model_name == "p1"


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("errors, expected", [
    ([("groq", StatusError(429)), ("openrouter", StatusError(503))], "busy right now"),
    ([("groq", StatusError(401)), ("openrouter", StatusError(403))], "API keys were rejected"),
])
def test_chain_failures_get_one_clear_message(errors, expected):
    assert expected in friendly_error(AllProvidersFailed(errors))
