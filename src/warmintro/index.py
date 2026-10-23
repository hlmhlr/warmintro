"""Turn a profile into searchable chunks and build the vector index.

Replaces the lab's data_processing.py. The lab dumped the whole profile as
one line of JSON and split it by size, so chunks started and ended in the
middle of JSON objects. Here each part of the profile (a job, a school, the
skills...) becomes its own readable Document, tagged with metadata.
"""

from dataclasses import dataclass, field

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from warmintro.config import CHUNK_OVERLAP, CHUNK_SIZE
from warmintro.llm import get_embeddings
from warmintro.profiles import Profile

OWNERS = ("them", "me")  # whose profile a chunk comes from


def _period(start: str | None, end: str | None) -> str:
    if not start:
        return ""
    return f" from {start} to {end or 'present'}"


def _sentence(*parts: str | None) -> str:
    """Join the non-empty parts with spaces."""
    return " ".join(part for part in parts if part)


def _details(*parts: str | None) -> str | None:
    """'(a, b)' from the non-empty parts, or None if there are none."""
    present = [part for part in parts if part]
    return f"({', '.join(present)})" if present else None


def profile_to_texts(profile: Profile) -> list[tuple[str, str]]:
    """Write each part of the profile as plain sentences: (section, text)."""
    name = profile.full_name
    texts = []

    place = ", ".join(p for p in (profile.city, profile.country) if p)
    texts.append(("overview", _sentence(
        f"{name}." if not profile.headline else f"{name}: {profile.headline}.",
        f"Based in {place}." if place else None,
        profile.summary,
    )))

    for job in profile.experiences:
        verb = "works" if job.starts_at and not job.ends_at else "worked"
        where = f" in {job.location}" if job.location else ""
        texts.append(("experience", _sentence(
            f"{name} {verb} as {job.title} at {job.company}{where}"
            f"{_period(job.starts_at, job.ends_at)}.",
            job.description,
        )))

    for school in profile.education:
        degree = ", ".join(p for p in (school.degree_name, school.field_of_study) if p)
        texts.append(("education", _sentence(
            f"{name} studied at {school.school}"
            f"{f' ({degree})' if degree else ''}"
            f"{_period(school.starts_at, school.ends_at)}.",
            school.description,
        )))

    for role in profile.volunteer_work:
        cause = f" (cause: {role.cause})" if role.cause else ""
        texts.append(("volunteering", _sentence(
            f"{name} volunteers as {role.role} at {role.organization}{cause}.",
            role.description,
        )))

    for project in profile.projects:
        texts.append(("project", _sentence(
            f"Project by {name}: {project.title}.", project.description)))

    if profile.skills:
        texts.append(("skills", f"Skills of {name}: {', '.join(profile.skills)}."))
    if profile.languages:
        texts.append(("languages", f"{name} speaks {', '.join(profile.languages)}."))
    if profile.certifications:
        items = [_sentence(c.name, _details(c.authority, c.year))
                 for c in profile.certifications]
        texts.append(("certifications", f"Certifications of {name}: {'; '.join(items)}."))
    if profile.awards:
        items = [_sentence(a.title, _details(a.issuer, a.year)) for a in profile.awards]
        texts.append(("awards", f"Awards and recognition of {name}: {'; '.join(items)}."))
    if profile.interests:
        texts.append(("interests", f"Interests of {name}: {', '.join(profile.interests)}."))

    return texts


def profile_to_documents(profile: Profile, owner: str = "them") -> list[Document]:
    """One LlamaIndex Document per profile section, with metadata.

    `owner` ("them" or "me") lets the common-ground mode keep two profiles in
    one index and filter by person. It is hidden from the embedding and the
    LLM, which only see the person's name and the section.
    """
    if owner not in OWNERS:
        raise ValueError(f"owner must be one of {OWNERS}")
    return [
        Document(
            text=text,
            metadata={"person": profile.full_name, "owner": owner, "section": section},
            excluded_embed_metadata_keys=["owner"],
            excluded_llm_metadata_keys=["owner"],
        )
        for section, text in profile_to_texts(profile)
    ]


def split_profile_data(documents: list[Document]) -> list[BaseNode]:
    """Split Documents into nodes. Most sections fit in one chunk."""
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.get_nodes_from_documents(documents)


def create_vector_database(
    nodes: list[BaseNode], embed_model: BaseEmbedding | None = None
) -> VectorStoreIndex:
    """Embed the nodes and store them in an in-memory vector index.

    Unlike the lab, errors are raised instead of returning None, so the
    caller can show a clear message (the lab logged the error and went on).
    """
    return VectorStoreIndex(nodes=nodes, embed_model=embed_model or get_embeddings())


def build_profile_index(
    profiles: dict[str, Profile], embed_model: BaseEmbedding | None = None
) -> VectorStoreIndex:
    """Load → split → index, for one or more profiles keyed by owner."""
    documents = []
    for owner, profile in profiles.items():
        documents += profile_to_documents(profile, owner)
    return create_vector_database(split_profile_data(documents), embed_model)


@dataclass
class EmbeddingReport:
    """Result of verify_embeddings."""

    nodes: int
    dimensions: set[int] = field(default_factory=set)
    missing: list[str] = field(default_factory=list)  # node ids with no vector
    empty: list[str] = field(default_factory=list)    # node ids with an all-zero vector

    @property
    def ok(self) -> bool:
        return self.nodes > 0 and not self.missing and not self.empty \
            and len(self.dimensions) == 1


def verify_embeddings(index: VectorStoreIndex) -> EmbeddingReport:
    """Check that every node in the index has a usable vector.

    The lab read the private index._storage_context and expected
    vector_store.get() to return None, but it raises KeyError, so missing
    vectors were never reported one by one.
    """
    node_ids = list(index.index_struct.nodes_dict)
    report = EmbeddingReport(nodes=len(node_ids))
    for node_id in node_ids:
        try:
            vector = index.vector_store.get(node_id)
        except KeyError:
            report.missing.append(node_id)
            continue
        report.dimensions.add(len(vector))
        if not any(vector):
            report.empty.append(node_id)
    return report
