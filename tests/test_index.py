import pytest

from fakes import KeywordEmbedding
from warmintro.index import (
    build_profile_index,
    profile_to_documents,
    profile_to_texts,
    split_profile_data,
    verify_embeddings,
)
from warmintro.profiles import CONTACTS_DIR, ME_DIR, load_profile, parse_profile


@pytest.fixture
def amara():
    return load_profile(CONTACTS_DIR / "amara_okafor.json")


def test_every_part_of_the_profile_becomes_a_section(amara):
    sections = [section for section, _ in profile_to_texts(amara)]
    assert sections.count("experience") == 3
    assert sections.count("education") == 2
    for name in ("overview", "volunteering", "project", "skills", "languages",
                 "certifications", "awards", "interests"):
        assert name in sections


def test_sections_are_readable_sentences(amara):
    texts = dict((s, t) for s, t in profile_to_texts(amara) if s != "experience")
    assert texts["skills"].startswith("Skills of Amara Okafor: Apache Kafka, Python")
    jobs = [t for s, t in profile_to_texts(amara) if s == "experience"]
    assert jobs[0].startswith("Current job: Amara Okafor works as Data Engineering Lead at "
                              "Brightwind Grid in Rotterdam, Netherlands from 2022-04 to present.")
    assert jobs[1].startswith("Past job: Amara Okafor worked as Senior Data Engineer")
    certs = dict(profile_to_texts(amara))["certifications"]
    assert "(Cloud Native Foundation, 2023)" in certs


def test_minimal_profile_has_only_an_overview():
    assert profile_to_texts(parse_profile({"full_name": "Kim"})) == [("overview", "Kim.")]


def test_documents_carry_metadata_hidden_from_the_model(amara):
    doc = profile_to_documents(amara, owner="me")[0]
    assert doc.metadata == {"person": "Amara Okafor", "owner": "me", "section": "overview"}
    assert "owner" in doc.excluded_embed_metadata_keys
    assert "owner" in doc.excluded_llm_metadata_keys


def test_unknown_owner_is_rejected(amara):
    with pytest.raises(ValueError):
        profile_to_documents(amara, owner="someone")


def test_each_section_fits_in_one_chunk(amara):
    documents = profile_to_documents(amara)
    nodes = split_profile_data(documents)
    assert len(nodes) == len(documents)
    assert all(node.metadata["section"] for node in nodes)


def test_index_holds_one_vector_per_node(amara):
    index = build_profile_index({"them": amara}, embed_model=KeywordEmbedding())
    report = verify_embeddings(index)
    assert report.ok
    assert report.nodes == len(profile_to_texts(amara))
    assert report.dimensions == {1024}


def test_retrieval_finds_the_right_section(amara):
    index = build_profile_index({"them": amara}, embed_model=KeywordEmbedding())
    top = index.as_retriever(similarity_top_k=1).retrieve("Which school did she study at?")
    assert top[0].node.metadata["section"] == "education"


def test_two_profiles_share_one_index():
    profiles = {"them": load_profile(CONTACTS_DIR / "amara_okafor.json"),
                "me": load_profile(ME_DIR / "alex_rivera.json")}
    index = build_profile_index(profiles, embed_model=KeywordEmbedding())
    owners = {node.metadata["owner"] for node in index.docstore.docs.values()}
    assert owners == {"them", "me"}
