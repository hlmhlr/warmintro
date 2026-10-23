import pytest

import warmintro.evaluate as ev
from fakes import KeywordEmbedding, ScriptedLLM
from warmintro.index import build_profile_index
from warmintro.profiles import CONTACTS_DIR, load_profile

GOOD_FACTS = """1. She leads data engineering at Brightwind Grid.
   What got you into energy?
2. She studied at Westmere Technical University.
   How was studying while working?
3. She maintains gridflow.
   How did gridflow start?"""


@pytest.mark.parametrize("text, expect, ok", [
    ("She is Data Engineering Lead.", ["data engineering lead"], True),
    ("She is a lead.", ["Data Engineering Lead"], False),
    ("No, she left in 2022.", [["no", "former"]], True),
    ("I know she works there.", [["no"]], False),          # "no" is not in "know"
    ("She played at a casino.", [["no"]], False),          # ...nor in "casino"
    ("About 40,000 people.", [["40,000", "40000"]], True),
    ("It is Orchard-Mini.", ["Orchard-Mini"], True),
])
def test_matches_uses_whole_words(text, expect, ok):
    assert ev.matches(text, expect) is ok


@pytest.mark.parametrize("text, ok", [
    ("I don't know.", True), ("I don’t know.", True), ("I do not know that.", True),
    ("She knows Python.", False),
])
def test_says_dont_know(text, ok):
    assert ev.says_dont_know(text) is ok


def test_facts_format():
    assert ev.check_facts_format(GOOD_FACTS) is None
    assert "text before" in ev.check_facts_format("Here are facts:\n" + GOOD_FACTS)
    assert "numbered" in ev.check_facts_format("1. A?\n2. B?")
    no_question = GOOD_FACTS.replace("How did gridflow start?", "")
    assert ev.check_facts_format(no_question) == "a fact has no question"


def test_unsupported_numbers_ignore_list_numbers():
    profile = "She joined in 2022 and has 6 engineers."
    assert ev.unsupported_numbers("1. Joined in 2022.\n2. Has 6 engineers.", profile) == []
    assert ev.unsupported_numbers("1. Volunteering since 2020.", profile) == ["2020"]


@pytest.mark.parametrize("reply, expected", [
    ("YES", (True, "")),
    ("No.\nsince 2020", (False, "since 2020")),
    ("NO", (False, "no reason given")),
])
def test_judge_reads_only_the_first_word(reply, expected):
    llm = ScriptedLLM(answers=[reply], prompts=[])
    assert ev.judge_statement("He volunteers since 2020.", "profile", llm) == expected


def test_judge_is_not_fooled_by_a_later_yes():
    # FaithfulnessEvaluator would pass this, because it contains "yes".
    llm = ScriptedLLM(answers=["Not sure, yes and no."], prompts=[])
    grounded, reason = ev.judge_statement("A statement.", "profile", llm)
    assert grounded is None and "unclear" in reason


def test_a_no_that_quotes_the_profile_is_unclear_not_a_failure():
    # Baseline: the judge answered NO and quoted a line of the profile.
    llm = ScriptedLLM(answers=["NO\nLukas Brenner worked as UX Researcher at Tandem Health"],
                      prompts=[])
    grounded, reason = ev.judge_statement("Lukas played cello for eight years.", "p", llm)
    assert grounded is None and "isn't in the statement" in reason


@pytest.mark.parametrize("fact, statement", [
    ("Lukas studied cello. How did that shape you?", "Lukas studied cello."),
    ('Amara leads a team. Conversation Starter: "What inspired you?"', "Amara leads a team."),
    ("Sofia has a PhD. - Friendly question: Why vision?", "Sofia has a PhD."),
    ("Amara maintains gridflow.\n   Question: How did it start?", "Amara maintains gridflow."),
    ("How are you?", ""),
])
def test_questions_are_removed_before_judging(fact, statement):
    assert ev.statement_only(fact) == statement


def test_facts_are_judged_one_at_a_time_without_questions():
    llm = ScriptedLLM(answers=["YES", "YES", "YES"], prompts=[])
    assert ev.judge_grounding(GOOD_FACTS, "profile", llm) == (True, "")
    assert len(llm.prompts) == 3
    statements = [p.split("Statement:\n")[1].split("\n\n")[0] for p in llm.prompts]
    assert statements[0] == "She leads data engineering at Brightwind Grid."
    assert not any("?" in statement for statement in statements)
    assert "She maintains gridflow." in llm.prompts[2]


def test_one_unsupported_fact_fails_the_facts():
    llm = ScriptedLLM(answers=["YES", "NO\nWestmere Technical University", "YES"], prompts=[])
    assert ev.judge_grounding(GOOD_FACTS, "profile", llm) == (
        False, "fact 2: Westmere Technical University")


def test_unclear_replies_are_reported_separately():
    llm = ScriptedLLM(answers=["YES", "maybe", "YES"], prompts=[])
    grounded, reason = ev.judge_grounding(GOOD_FACTS, "profile", llm)
    assert grounded is None and reason.startswith("fact 2: unclear")


@pytest.fixture
def amara_index():
    return build_profile_index({"them": load_profile(CONTACTS_DIR / "amara_okafor.json")},
                               embed_model=KeywordEmbedding())


def test_question_kinds_are_scored(amara_index):
    questions = {
        "answerable": [{"question": "Job?", "expect": ["Lead"]},
                       {"question": "School?", "expect": ["Westmere"]}],
        "unanswerable": ["Salary?", "Children?"],
        "qualifiers": [{"question": "Dutch?", "expect": ["intermediate"]}],
    }
    llm = ScriptedLLM(prompts=[], answers=[
        "Data Engineering Lead.", "I don't know.",       # answerable: pass, fail
        "I don't know.", "About 5000 euros.",            # unanswerable: pass, fail
        "She speaks Dutch.",                             # qualifier: fail
    ])
    results = ev.evaluate_questions("amara", amara_index, questions, llm=llm)
    assert [(r.kind, r.passed) for r in results] == [
        ("answerable", True), ("answerable", False),
        ("unanswerable", True), ("unanswerable", False), ("qualifier", False)]
    assert results[1].detail == "said 'I don't know'"


def test_facts_checks_and_judge(amara_index):
    profile = load_profile(CONTACTS_DIR / "amara_okafor.json")
    llm = ScriptedLLM(answers=[GOOD_FACTS, "Intro\n" + GOOD_FACTS + " since 1999"], prompts=[])
    judge = ScriptedLLM(answers=["YES", "YES", "YES", "YES", "YES", "NO\ngridflow since 1999"],
                        prompts=[])
    good, bad = ev.evaluate_facts("amara", amara_index, profile, runs=2, llm=llm,
                                  judge_llm=judge)
    assert good.checks["format"] and good.checks["numbers"] and good.checks["judge"]
    assert not bad.checks["format"] and bad.checks["numbers"] is False
    assert bad.checks["judge"] is False and "1999" in bad.detail
    # The judge sees the whole profile, not only the retrieved chunks.
    assert "Amara Okafor speaks English" in judge.prompts[0]


def test_scorecard_table_and_failures(amara_index):
    results = [
        ev.Result("amara", "answerable", "Job?", "Lead", True, seconds=2.0),
        ev.Result("amara", "unanswerable", "Salary?", "5000", False, "answered", 4.0),
        ev.Result("amara", "facts", "facts run 1", "x", False, "judge: no", 10.0,
                  checks={"format": True, "numbers": True, "used_job": False, "judge": False}),
    ]
    results[2].sections = ["skills", "education"]
    report = ev.scorecard(results, "baseline", "LLM: fake")
    assert "| Facts: judge reply unclear (not counted) | 0/1 | 0/1 |" in report
    assert "Sections: skills, education" in report
    assert "| Answerable questions correct | 1/1 | 1/1 |" in report
    assert "| Unanswerable → \"I don't know.\" | 0/1 | 0/1 |" in report
    assert "| Facts: a job was in the context | 0/1 | 0/1 |" in report
    assert "| Seconds per answer | 3.0 s | 3.0 s |" in report
    assert "## Failures (2)" in report and "Salary?" in report


def test_shipped_question_set_is_complete():
    questions = ev.load_questions()
    assert set(questions) == {"amara_okafor", "lukas_brenner", "sofia_marchetti"}
    for name, q in questions.items():
        assert (CONTACTS_DIR / f"{name}.json").exists()
        assert len(q["answerable"]) == 6 and len(q["unanswerable"]) == 4
        assert len(q["qualifiers"]) == 2


def test_expected_words_are_really_in_the_profiles():
    # A question whose expected answer isn't in the profile could never pass.
    for name, q in ev.load_questions().items():
        text = "\n".join(t for _, t in ev.profile_to_texts(
            load_profile(CONTACTS_DIR / f"{name}.json")))
        for item in q["answerable"] + [i for i in q["qualifiers"]
                                       if not isinstance(i["expect"][0], list)]:
            assert ev.matches(text, item["expect"]), (name, item)


def test_rejudge_replaces_only_the_judge_verdict():
    profiles = {"amara": load_profile(CONTACTS_DIR / "amara_okafor.json")}
    old = ev.Result("amara", "facts", "facts run 1", GOOD_FACTS, False,
                    "text before fact 1; judge: old verdict",
                    checks={"format": False, "numbers": True, "used_job": True, "judge": False})
    answer = ev.Result("amara", "answerable", "Job?", "Lead", True)
    ev.rejudge([old, answer], profiles, ScriptedLLM(answers=["YES"], prompts=[]))
    assert old.checks["judge"] is True
    assert old.detail == "text before fact 1"   # rule checks are kept
    assert old.passed is False
    assert answer.passed is True                # answers are untouched


def test_results_round_trip(tmp_path):
    results = [ev.Result("amara", "facts", "run", "text", True, checks={"judge": None})]
    ev.save_results(results, tmp_path / "r.json")
    assert ev.load_results(tmp_path / "r.json") == results


@pytest.fixture
def sample_profiles():
    return {name: load_profile(CONTACTS_DIR / f"{name}.json")
            for name in ("amara_okafor", "lukas_brenner", "sofia_marchetti")}


def test_calibration_set_is_balanced_and_true_items_are_really_supported():
    import json
    items = json.loads(ev.CALIBRATION_PATH.read_text(encoding="utf-8"))["statements"]
    assert sum(i["supported"] for i in items) == sum(not i["supported"] for i in items) == 6
    assert all("error" in i for i in items if not i["supported"])


def test_a_judge_that_always_says_yes_scores_50_percent(sample_profiles):
    c = ev.calibrate_judge(ScriptedLLM(answers=["YES"], prompts=[]), sample_profiles)
    assert (c.true_accepted, c.false_rejected, c.unclear) == (6, 0, 0)
    assert c.accuracy == 0.5 and len(c.misses) == 6


def test_a_perfect_judge_scores_100_percent(sample_profiles):
    import json
    items = json.loads(ev.CALIBRATION_PATH.read_text(encoding="utf-8"))["statements"]
    # Each NO quotes the statement's own words, so it is not marked unclear.
    replies = ["YES" if i["supported"] else "NO\n" + i["statement"] for i in items]
    c = ev.calibrate_judge(ScriptedLLM(answers=replies, prompts=[]), sample_profiles)
    assert c.accuracy == 1.0 and c.misses == []
