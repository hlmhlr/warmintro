import pytest

from fakes import KeywordEmbedding, ScriptedLLM
from warmintro.common_ground import (
    Pair,
    check_shared_claims,
    find_common_ground,
    limit_items,
    match_chunks,
    rank_pairs,
    shared_terms,
)
from warmintro.query import Source
from warmintro.index import build_profile_index
from warmintro.profiles import CONTACTS_DIR, ME_DIR, load_profile


@pytest.fixture
def alex():
    return load_profile(ME_DIR / "alex_rivera.json")


@pytest.fixture
def amara():
    return load_profile(CONTACTS_DIR / "amara_okafor.json")


def test_pairs_always_match_my_chunk_with_theirs(alex, amara):
    index = build_profile_index({"me": alex, "them": amara}, embed_model=KeywordEmbedding())
    pairs = match_chunks(index)
    assert pairs
    # Without the owner filter, a chunk would match itself with score 1.0.
    assert all(p.mine.person == "Alex Rivera" for p in pairs)
    assert all(p.theirs.person == "Amara Okafor" for p in pairs)
    assert all(p.score < 0.99 for p in pairs)


def test_pairs_are_ordered_and_their_chunks_are_not_reused(alex, amara):
    index = build_profile_index({"me": alex, "them": amara}, embed_model=KeywordEmbedding())
    pairs = match_chunks(index, top_pairs=5)
    keys = [(bool(p.shared), p.score) for p in pairs]
    assert keys == sorted(keys, reverse=True) and len(pairs) == 5
    assert len({p.theirs.text for p in pairs}) == len(pairs)


def test_shared_school_is_found_and_ranked_first(alex, amara):
    index = build_profile_index({"me": alex, "them": amara}, embed_model=KeywordEmbedding())
    pairs = match_chunks(index)
    school = next(p for p in pairs if "Westmere" in p.mine.text and "Westmere" in p.theirs.text)
    assert "Westmere" in school.shared
    # Pairs with shared words come before pairs with none.
    has_words = [bool(p.shared) for p in pairs]
    assert has_words == sorted(has_words, reverse=True)


def test_no_extra_embedding_calls(alex, amara):
    embed = KeywordEmbedding(seen=[], queries=[])
    index = build_profile_index({"me": alex, "them": amara}, embed_model=embed)
    match_chunks(index)
    assert embed.queries == []  # stored vectors are reused as queries


def test_prompt_labels_both_people(alex, amara):
    llm = ScriptedLLM(answers=["1. Shared: Westmere.\n   Question: Which year?"], prompts=[])
    result = find_common_ground(alex, amara, llm=llm, embed_model=KeywordEmbedding())
    prompt = llm.prompts[0]
    assert "Alex Rivera start a conversation with Amara Okafor" in prompt
    assert prompt.count("A: Alex Rivera") + prompt.count("A: Current job: Alex") \
        + prompt.count("A: Past job: Alex") + prompt.count("A: Skills of Alex") >= 1
    assert "No clear common ground found." in prompt
    assert "Words in both: Westmere" in prompt
    assert result.text.startswith("1. Shared: Westmere.")
    assert len(result.pairs) == 5


NAMES = {"alex", "rivera", "amara", "okafor"}


@pytest.mark.parametrize("a, b, expected", [
    ("Alex Rivera studied at Westmere Technical University (MSc, Data Science).",
     "Amara Okafor studied at Westmere Technical University (MSc, Energy).",
     ["Westmere", "Technical", "MSc"]),
    ("Interests of Alex Rivera: Half marathons.", "Interests of Amara Okafor: Marathon running.",
     ["marathons"]),                                   # plural and singular match
    ("Skills of Alex Rivera: Python, SQL.", "Skills of Amara Okafor: Kafka, dbt.", []),
])
def test_shared_terms_skip_names_and_sentence_patterns(a, b, expected):
    assert shared_terms(a, b, NAMES) == expected


def test_claims_only_one_person_has_are_flagged():
    sam = load_profile(ME_DIR / "sam_lindqvist.json")
    lukas = load_profile(CONTACTS_DIR / "lukas_brenner.json")
    # Real replies from a run: Lukas doesn't speak Swedish, and wasn't amateur.
    text = ("1. Shared: Altstadt Conservatory of Music\n   Question: x?\n"
            "2. Shared: German, Swedish, English\n   Question: y?\n"
            "3. Shared: Cello in an amateur orchestra\n   Question: z?")
    checks = check_shared_claims(text, sam, lukas)
    assert [c.supported for c in checks] == [True, False, False]
    assert checks[1].one_sided == ["Swedish"] and checks[2].one_sided == ["amateur"]


def test_model_wording_in_neither_profile_is_ignored(alex, amara):
    [check] = check_shared_claims("1. Shared: Python programming", alex, amara)
    assert check.supported


def test_reply_is_cut_to_three_icebreakers():
    text = "1. Shared: a\n   Question: ?\n2. B\n3. C\n4. Shared: d\n   Question: ?"
    assert limit_items(text, 3) == "1. Shared: a\n   Question: ?\n2. B\n3. C"
    assert limit_items("1. A\n2. B", 3) == "1. A\n2. B"


def _pair(their_text, score, shared):
    mine = Source("Me", "skills", 1.0, "mine")
    return Pair(mine, Source("Them", "skills", score, their_text), score, shared)


def test_shared_words_beat_a_higher_similarity():
    # Real run: "Skills of A" vs "Skills of B" scored 0.78 with nothing shared,
    # while the shared school scored lower.
    same_pattern = _pair("Skills of Amara: Kafka", 0.78, [])
    school = _pair("Amara studied at Westmere", 0.70, ["Westmere"])
    assert rank_pairs([same_pattern, school]) == [school, same_pattern]


def test_each_of_their_chunks_is_used_once():
    a, b = _pair("same chunk", 0.9, ["x"]), _pair("same chunk", 0.8, ["y"])
    assert rank_pairs([a, b]) == [a]


def test_find_common_ground_keeps_three_icebreakers(alex, amara):
    reply = "\n".join(f"{n}. Shared: thing {n}\n   Question: why?" for n in range(1, 5))
    result = find_common_ground(alex, amara, llm=ScriptedLLM(answers=[reply], prompts=[]),
                                embed_model=KeywordEmbedding())
    assert "3. Shared" in result.text and "4. Shared" not in result.text
