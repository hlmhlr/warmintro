"""Regression tests for problems in the lab's data processing code."""

from llama_index.core.schema import MetadataMode

from fakes import KeywordEmbedding
from warmintro.index import (
    build_profile_index,
    profile_to_documents,
    split_profile_data,
    verify_embeddings,
)
from warmintro.profiles import CONTACTS_DIR, load_profile, parse_profile


def amara():
    return load_profile(CONTACTS_DIR / "amara_okafor.json")


def test_whole_chunks_are_embedded_not_the_first_tokens():
    # Lab: truncate_input_tokens=3 embedded only 3 tokens of each chunk.
    embed = KeywordEmbedding(seen=[])
    index = build_profile_index({"them": amara()}, embed_model=embed)
    chunks = [n.get_content(MetadataMode.EMBED) for n in index.docstore.docs.values()]
    assert sorted(embed.seen) == sorted(chunks)
    assert min(len(text.split()) for text in embed.seen) > 3


def test_chunks_never_start_inside_json():
    # Lab: json.dumps(profile) split by size, so chunks began with '"}, {'.
    for node in split_profile_data(profile_to_documents(amara())):
        text = node.get_content()
        assert text[0].isalpha() and "{" not in text and '"' not in text


def test_long_section_uses_a_small_overlap():
    # Lab: default overlap of 200 tokens on 400-token chunks. A section of
    # 50 sentences fits in 2 chunks with overlap 40 but needs 3 with 200.
    description = " ".join(
        f"Sentence number {i} describes one more detail of the job." for i in range(50))
    profile = parse_profile({"full_name": "Kim", "experiences": [
        {"title": "Dev", "company": "X", "description": description}]})
    nodes = split_profile_data(profile_to_documents(profile))
    assert [n.metadata["section"] for n in nodes].count("experience") == 2


def test_missing_vector_is_reported_by_node_id():
    # Lab: vector_store.get() raises KeyError, so its "is None" check never ran.
    index = build_profile_index({"them": amara()}, embed_model=KeywordEmbedding())
    node_id = next(iter(index.index_struct.nodes_dict))
    del index.vector_store.data.embedding_dict[node_id]
    report = verify_embeddings(index)
    assert report.missing == [node_id]
    assert not report.ok


def test_all_zero_vector_is_reported():
    index = build_profile_index({"them": amara()}, embed_model=KeywordEmbedding())
    node_id = next(iter(index.index_struct.nodes_dict))
    index.vector_store.data.embedding_dict[node_id] = [0.0] * 1024
    report = verify_embeddings(index)
    assert report.empty == [node_id]
    assert not report.ok


def test_mixed_vector_sizes_are_reported():
    index = build_profile_index({"them": amara()}, embed_model=KeywordEmbedding())
    node_id = next(iter(index.index_struct.nodes_dict))
    index.vector_store.data.embedding_dict[node_id] = [1.0] * 10
    assert not verify_embeddings(index).ok


def test_nested_empty_fields_do_not_reach_the_model():
    # Lab: only top-level empty values were removed.
    texts = [n.get_content() for n in split_profile_data(profile_to_documents(amara()))]
    junior = next(t for t in texts if "Junior Developer" in t)
    assert junior.endswith("to 2017-08.")  # no trailing empty description
