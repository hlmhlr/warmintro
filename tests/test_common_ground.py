import pytest

from fakes import KeywordEmbedding, ScriptedLLM
from warmintro.common_ground import find_common_ground, match_chunks
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


def test_best_pairs_come_first_and_their_chunks_are_not_reused(alex, amara):
    index = build_profile_index({"me": alex, "them": amara}, embed_model=KeywordEmbedding())
    pairs = match_chunks(index, top_pairs=5)
    scores = [p.score for p in pairs]
    assert scores == sorted(scores, reverse=True) and len(pairs) == 5
    assert len({p.theirs.text for p in pairs}) == len(pairs)


def test_shared_school_is_found(alex, amara):
    index = build_profile_index({"me": alex, "them": amara}, embed_model=KeywordEmbedding())
    pairs = match_chunks(index)
    assert any("Westmere" in p.mine.text and "Westmere" in p.theirs.text for p in pairs)


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
    assert result.text.startswith("1. Shared: Westmere.")
    assert len(result.pairs) == 5
