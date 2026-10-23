import pytest

import warmintro.common_ground
import warmintro.index
import warmintro.query
import warmintro.webapp as web
from fakes import KeywordEmbedding, ScriptedLLM


MODELS_ASKED = []  # model names the app asked for


@pytest.fixture
def llm(monkeypatch, clean_env):
    """Fake models everywhere the app reaches them; records the model names asked for."""
    fake = ScriptedLLM(answers=["1. Fact.\n   Question: Why?"], prompts=[])
    MODELS_ASKED.clear()

    def get_llm(generation, model=None):
        MODELS_ASKED.append(model)
        return fake

    monkeypatch.setattr(warmintro.index, "get_embeddings", lambda settings=None: KeywordEmbedding())
    for module in (warmintro.query, warmintro.common_ground, web):
        monkeypatch.setattr(module, "get_llm", get_llm)
    return fake


def test_prepare_builds_a_session_with_its_own_model(llm):
    facts, common, session, chat = web.prepare("amara_okafor", "", web.NONE, "", "granite4:micro-h")
    assert facts.startswith("### 3 facts about Amara Okafor") and "1. Fact." in facts
    assert common == "" and chat == []
    assert session["name"] == "Amara Okafor" and session["model"] == "granite4:micro-h"
    assert MODELS_ASKED == ["granite4:micro-h"]  # no global model switch (lab bug)


def test_prepare_with_me_adds_common_ground(llm):
    llm.answers = ["1. Fact.", "1. Shared: Westmere.\n   Question: Which year?"]
    _, common, session, _ = web.prepare("amara_okafor", "", "alex_rivera", "", "m")
    assert common.startswith("### What you have in common") and "Westmere" in common


def test_pasted_profiles_work_and_contact_details_are_removed(llm):
    facts, _, session, _ = web.prepare(
        web.PASTE, "Kim builds compilers in Oslo. Mail kim@example.com", web.NONE, "", "m")
    assert "### 3 facts about This person" in facts
    assert "kim@example.com" not in llm.prompts[0] and "[removed]" in llm.prompts[0]


def test_prepare_errors_become_a_message_not_a_traceback(llm):
    llm.answers = [ConnectionError("Ollama is down")]
    facts, common, session, chat = web.prepare("amara_okafor", "", web.NONE, "", "m")
    assert facts.startswith("⚠️") and "Is Ollama running?" in facts
    assert session is None and "Traceback" not in facts


def test_bad_paste_gets_its_own_message(llm):
    facts, *_ = web.prepare(web.PASTE, "   ", web.NONE, "", "m")
    assert "Paste a profile first." in facts


def test_ask_appends_question_and_answer(llm):
    *_, session, _ = web.prepare("amara_okafor", "", web.NONE, "", "m")
    llm.answers, llm.prompts = ["She studied at Westmere."], []
    box, history = web.ask("Where did she study?", [], session)
    assert box == ""
    assert history == [{"role": "user", "content": "Where did she study?"},
                       {"role": "assistant", "content": "She studied at Westmere."}]
    assert session["questions"] == 1


def test_ask_before_loading_a_profile(llm):
    _, history = web.ask("Hello?", [], None)
    assert "Load a profile first" in history[-1]["content"]


def test_empty_question_changes_nothing(llm):
    assert web.ask("  ", [{"role": "user", "content": "x"}], None) == \
        ("", [{"role": "user", "content": "x"}])


def test_question_limit_per_session(llm, monkeypatch):
    monkeypatch.setenv("MAX_QUESTIONS_PER_SESSION", "1")
    *_, session, _ = web.prepare("amara_okafor", "", web.NONE, "", "m")
    web.ask("First?", [], session)
    _, history = web.ask("Second?", [], session)
    assert "allows 1 questions" in history[-1]["content"]


def test_answer_errors_stay_in_the_chat(llm):
    *_, session, _ = web.prepare("amara_okafor", "", web.NONE, "", "m")
    llm.answers, llm.prompts = [RuntimeError("secret /srv path")], []
    _, history = web.ask("Where did she study?", [], session)
    assert history[-1]["content"].startswith("⚠️") and "secret" not in history[-1]["content"]


def test_choices_list_samples_and_paste():
    assert ("Amara Okafor (sample)", "amara_okafor") in web.contact_choices()
    assert web.contact_choices()[-1][1] == web.PASTE
    assert web.me_choices()[0][1] == web.NONE


def test_gemini_offers_only_its_model(clean_env):
    clean_env.setenv("LLM_PROVIDER", "gemini")
    assert web.available_models() == [("gemini-flash-latest", "gemini-flash-latest")]


def test_ollama_down_still_offers_the_configured_model(clean_env):
    clean_env.setenv("OLLAMA_BASE_URL", "http://localhost:1")
    assert web.available_models() == [("granite4:tiny-h", "granite4:tiny-h")]


def test_a_chain_offers_one_automatic_choice(clean_env):
    clean_env.setenv("LLM_PROVIDER", "groq,cloudflare")
    [(label, value)] = web.available_models()
    assert value == web.AUTO and label.startswith("automatic: openai/gpt-oss-120b (groq) →")


def test_automatic_choice_lets_each_provider_use_its_model(llm):
    web.prepare("amara_okafor", "", web.NONE, "", web.AUTO)
    assert MODELS_ASKED == [None]


def test_the_provider_that_answered_is_shown(llm, monkeypatch):
    from warmintro.fallback import FallbackLLM
    from fakes import ScriptedLLM as Fake

    chain = FallbackLLM(llms=[Fake(answers=[RuntimeError("429")], prompts=[]),
                              Fake(answers=["1. Fact."], prompts=[])],
                        names=["groq", "cloudflare"])
    monkeypatch.setattr(web, "get_llm", lambda generation, model=None: chain)
    facts, _, session, _ = web.prepare("amara_okafor", "", web.NONE, "", web.AUTO)
    assert facts.endswith("<sub>Written by cloudflare</sub>")
    _, history = web.ask("Where did she study?", [], session)
    assert history[-1]["content"].endswith("*(cloudflare)*")


def test_server_settings(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert web.server_settings() == {"server_name": "127.0.0.1", "server_port": 7860}
    monkeypatch.setenv("PORT", "10000")
    assert web.server_settings() == {"server_name": "0.0.0.0", "server_port": 10000}


def test_ui_builds_with_named_api_events(clean_env):
    clean_env.setenv("OLLAMA_BASE_URL", "http://localhost:1")
    demo = web.build_ui()
    names = {dep.get("api_name") for dep in demo.config["dependencies"]}
    assert {"icebreakers", "ask"} <= names


def test_startup_stops_when_the_provider_packages_are_missing(clean_env):
    import sys
    clean_env.setenv("LLM_PROVIDER", "gemini")
    clean_env.setenv("GOOGLE_API_KEY", "fake-key")
    clean_env.setitem(sys.modules, "llama_index.llms.google_genai", None)
    with pytest.raises(SystemExit, match="uv sync --extra gemini") as stop:
        web.check_models()
    assert "LLM: gemini-flash-latest (gemini)" in str(stop.value)


def test_startup_check_passes_for_the_local_setup(clean_env):
    web.check_models()  # Ollama models are created without contacting Ollama
