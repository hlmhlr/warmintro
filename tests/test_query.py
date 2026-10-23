import pytest

import warmintro.query as query
from fakes import KeywordEmbedding, ScriptedLLM
from warmintro.config import ANSWER_GENERATION, FACTS_GENERATION, SIMILARITY_TOP_K
from warmintro.index import build_profile_index
from warmintro.profiles import CONTACTS_DIR, load_profile


@pytest.fixture
def embed():
    return KeywordEmbedding(seen=[], queries=[])


@pytest.fixture
def index(embed):
    return build_profile_index(
        {"them": load_profile(CONTACTS_DIR / "amara_okafor.json")}, embed_model=embed)


def test_facts_use_the_facts_template_and_the_whole_profile(index):
    llm = ScriptedLLM(answers=["1. Fact one.\n2. Fact two.\n3. Fact three."], prompts=[])
    answer = query.generate_initial_facts(index, llm=llm)
    assert answer.text.startswith("1. Fact one.")
    assert len(llm.prompts) == 1
    assert "networking event" in llm.prompts[0]
    assert "career or education" in llm.prompts[0]
    # Every chunk, so the jobs are always there (baseline: no job in 6 of 9 runs).
    assert len(answer.sources) == len(index.index_struct.nodes_dict)
    assert "experience" in answer.sections


def test_answer_uses_the_question_template_and_reports_sources(index):
    llm = ScriptedLLM(answers=["  She studied at Westmere Technical University.  "], prompts=[])
    answer = query.answer_user_query(index, "Where did she study?", llm=llm)
    assert answer.text == "She studied at Westmere Technical University."  # stripped
    prompt = llm.prompts[0]
    assert "I don't know." in prompt and "Question: Where did she study?" in prompt
    assert "Amara Okafor studied at" in prompt  # the retrieved chunks are in it
    assert len(answer.sources) == SIMILARITY_TOP_K
    assert answer.sections[0] == "education"
    assert answer.sources[0].person == "Amara Okafor"


def test_each_question_is_retrieved_once(index, embed):
    # Lab: retrieved once by hand, then again inside the query engine.
    query.answer_user_query(index, "Where did she study?", llm=ScriptedLLM(prompts=[]))
    assert embed.queries == ["Where did she study?"]


def test_the_owner_tag_never_reaches_the_llm(index):
    llm = ScriptedLLM(prompts=[])
    query.answer_user_query(index, "Which languages does she speak?", llm=llm)
    assert "owner" not in llm.prompts[0]
    assert "section: languages" in llm.prompts[0]


def test_llm_errors_are_raised_not_returned_as_text(index):
    # Lab: returned "Failed to get an answer." and callers crashed on .response.
    llm = ScriptedLLM(answers=[ConnectionError("Ollama is down")], prompts=[])
    with pytest.raises(ConnectionError):
        query.answer_user_query(index, "Where did she study?", llm=llm)


def test_empty_question_is_rejected(index):
    with pytest.raises(ValueError):
        query.answer_user_query(index, "   ", llm=ScriptedLLM(prompts=[]))


@pytest.mark.parametrize("function, generation", [
    (query.generate_initial_facts, FACTS_GENERATION),
    (query.answer_user_query, ANSWER_GENERATION),
])
def test_each_task_gets_its_own_settings_and_model(index, monkeypatch, function, generation):
    calls = []

    def fake_get_llm(gen, model=None):
        calls.append((gen, model))
        return ScriptedLLM(prompts=[])

    monkeypatch.setattr(query, "get_llm", fake_get_llm)
    args = (index,) if function is query.generate_initial_facts else (index, "Hi?")
    function(*args, model="granite4:micro-h")
    assert calls == [(generation, "granite4:micro-h")]


def test_facts_are_capped_for_very_long_profiles(index, monkeypatch):
    monkeypatch.setattr(query, "FACTS_MAX_CHUNKS", 5)
    answer = query.generate_initial_facts(index, llm=ScriptedLLM(prompts=[]))
    assert len(answer.sources) == 5
