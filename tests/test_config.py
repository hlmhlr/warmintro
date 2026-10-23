import pytest

from warmintro import config
from warmintro.config import Generation, load_settings


def test_defaults_are_local(clean_env):
    s = load_settings()
    assert (s.llm_provider, s.embeddings_provider) == ("ollama", "ollama")
    assert s.ollama_model == "granite4:tiny-h"
    assert s.ollama_embed_model == "granite-embedding:278m"


def test_env_overrides_and_is_case_insensitive(clean_env):
    clean_env.setenv("LLM_PROVIDER", " Gemini ")
    clean_env.setenv("EMBEDDINGS_PROVIDER", "huggingface")
    s = load_settings()
    assert (s.llm_provider, s.embeddings_provider) == ("gemini", "huggingface")


@pytest.mark.parametrize("name", ["LLM_PROVIDER", "EMBEDDINGS_PROVIDER"])
def test_unknown_provider_lists_the_choices(clean_env, name):
    clean_env.setenv(name, "watsonx")
    with pytest.raises(ValueError, match="Choose one of: ollama"):
        load_settings()


def test_greedy_ignores_temperature():
    assert Generation(100, "greedy", temperature=0.9).options() == {
        "temperature": 0.0, "top_k": 1, "top_p": 1.0, "max_new_tokens": 100,
    }


def test_sampling_at_temperature_zero_is_rejected():
    # The lab's facts call used decoding_method="sample" with temperature 0.0.
    with pytest.raises(ValueError, match="temperature above 0"):
        Generation(500, "sample", temperature=0.0)


def test_unknown_decoding_method_is_rejected():
    with pytest.raises(ValueError):
        Generation(100, "beam")


def test_presets():
    assert config.FACTS_GENERATION.options()["temperature"] > 0
    assert config.ANSWER_GENERATION.options()["temperature"] == 0


@pytest.mark.parametrize(
    "template", [config.INITIAL_FACTS_TEMPLATE, config.USER_QUESTION_TEMPLATE]
)
def test_templates_have_llamaindex_slots(template):
    assert "{context_str}" in template and "{query_str}" in template
    assert "only" in template  # answers come from the profile only


def test_question_template_admits_missing_answers():
    assert "I don't know" in config.USER_QUESTION_TEMPLATE


def test_facts_template_rules_from_the_baseline():
    t = config.INITIAL_FACTS_TEMPLATE
    assert 'Start your reply with "1."' in t          # 6/9 baseline runs had an intro
    assert "never add new ones" in t                  # invented dates
    assert "Do not add opinions" in t                 # "demonstrating her dedication..."
    assert "include at least" in t and "one job" in t
    assert t.count("Question: <question>") == 3       # one fixed layout


def test_question_template_keeps_qualifiers():
    assert '"basic"' in config.USER_QUESTION_TEMPLATE  # "basic" Italian was dropped
    assert "current or past" in config.USER_QUESTION_TEMPLATE
