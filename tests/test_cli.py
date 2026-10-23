import pytest

import warmintro.cli as cli
import warmintro.common_ground
import warmintro.index
import warmintro.query
from fakes import KeywordEmbedding, ScriptedLLM
from warmintro.profiles import CONTACTS_DIR


def scripted_inputs(*lines):
    """An input() replacement that returns the lines, then exits."""
    queue = list(lines)
    return lambda prompt="": queue.pop(0) if queue else "exit"


@pytest.fixture
def fake_models(monkeypatch):
    """Fake embeddings and LLM everywhere the CLI reaches them."""
    llm = ScriptedLLM(answers=["1. Fact.\n2. Fact.\n3. Fact.", "She studied in Lagos."],
                      prompts=[])
    monkeypatch.setattr(warmintro.index, "get_embeddings", lambda settings=None: KeywordEmbedding())
    monkeypatch.setattr(warmintro.query, "get_llm", lambda generation, model=None: llm)
    monkeypatch.setattr(warmintro.common_ground, "get_llm", lambda generation, model=None: llm)
    return llm


def test_resolve_profile_accepts_a_sample_name_or_a_path():
    assert cli.resolve_profile("lukas_brenner") == CONTACTS_DIR / "lukas_brenner.json"
    path = CONTACTS_DIR / "sofia_marchetti.json"
    assert cli.resolve_profile(str(path)) == path


def test_unknown_profile_lists_the_samples():
    with pytest.raises(SystemExit, match="amara_okafor, lukas_brenner"):
        cli.resolve_profile("nobody")


def test_choose_profile_defaults_to_the_first(capsys):
    assert cli.choose_profile(scripted_inputs("")).name == "amara_okafor.json"
    assert cli.choose_profile(scripted_inputs("3")).name == "sofia_marchetti.json"
    with pytest.raises(SystemExit):
        cli.choose_profile(scripted_inputs("9"))


def test_full_cli_run(fake_models, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", scripted_inputs("Where did she study?", "", "bye"))
    cli.main(["--profile", "amara_okafor"])
    out = capsys.readouterr().out
    assert "Here are 3 interesting facts about Amara Okafor:" in out
    assert "1. Fact." in out
    assert "Bot: She studied in Lagos." in out
    assert out.rstrip().endswith("Bot: Goodbye!")
    assert len(fake_models.prompts) == 2  # facts + one question; the empty line is skipped


def test_chat_survives_a_failed_answer(fake_models, monkeypatch, capsys):
    fake_models.answers = ["facts", ConnectionError("down"), "Second try works."]
    monkeypatch.setattr("builtins.input", scripted_inputs("Q1?", "Q2?"))
    cli.main(["--profile", "amara_okafor"])
    out = capsys.readouterr().out
    assert "Is Ollama running?" in out
    assert "Bot: Second try works." in out
    assert "Traceback" not in out


@pytest.mark.parametrize("key", [EOFError, KeyboardInterrupt])
def test_ctrl_c_or_end_of_input_exits_cleanly(fake_models, monkeypatch, capsys, key):
    def interrupted(prompt=""):
        raise key
    monkeypatch.setattr("builtins.input", interrupted)
    cli.main(["--profile", "amara_okafor"])
    assert "Goodbye!" in capsys.readouterr().out


def test_startup_failure_exits_with_a_friendly_message(fake_models):
    fake_models.answers = [ConnectionError("down")]
    with pytest.raises(SystemExit, match="Is Ollama running"):
        cli.main(["--profile", "amara_okafor"])


def test_me_option_adds_common_ground(fake_models, monkeypatch, capsys):
    fake_models.answers = ["1. Fact.", "1. Shared: Westmere Technical University.", "Bye"]
    monkeypatch.setattr("builtins.input", scripted_inputs())
    cli.main(["--profile", "amara_okafor", "--me", "alex_rivera"])
    out = capsys.readouterr().out
    assert "What you have in common with Amara Okafor:" in out
    assert "Shared: Westmere Technical University." in out
    assert "A: " in fake_models.prompts[1] and "Alex Rivera" in fake_models.prompts[1]


def test_unknown_me_profile_lists_my_samples():
    with pytest.raises(SystemExit, match="alex_rivera, sam_lindqvist"):
        cli.main(["--profile", "amara_okafor", "--me", "nobody"])
