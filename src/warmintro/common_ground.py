"""Find what you and another person share, and turn it into icebreakers.

Both profiles go into one index, tagged owner="me" or owner="them". Each of
your chunks is used as a query against their chunks only, through a metadata
filter, reusing the vectors already stored (no extra embedding calls). The
best-matching pairs go to the LLM, which keeps only what is really shared.
"""

from dataclasses import dataclass

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from warmintro.config import FACTS_GENERATION
from warmintro.index import build_profile_index
from warmintro.llm import get_llm
from warmintro.profiles import Profile
from warmintro.query import Source

TOP_PAIRS = 5        # pairs sent to the LLM
MATCHES_PER_CHUNK = 2  # people often have two schools or several jobs

COMMON_GROUND_TEMPLATE = PromptTemplate("""You help {me} start a conversation with {them} at a networking event.
Below are pairs of profile excerpts that may have something in common.
In each pair, line A is about {me} and line B is about {them}.

{pairs}

Write up to three icebreakers about things both people really share, such as
the same school, city, skill, hobby, cause or career change.

Rules:
- Start your reply with "1." Write no introduction and no closing line.
- Use only facts from the excerpts. Skip a pair that shares nothing.
- Each icebreaker names what they share, then gives one question {me} could
  ask {them}, using "you".
- If nothing is shared, reply exactly: No clear common ground found.

Use this layout:
1. Shared: <what both have in common>
   Question: <question>""")


@dataclass
class Pair:
    mine: Source
    theirs: Source
    score: float


@dataclass
class CommonGround:
    text: str
    pairs: list[Pair]


def _source(node, score: float) -> Source:
    return Source(person=node.metadata["person"], section=node.metadata["section"],
                  score=score, text=node.get_content())


def match_chunks(index: VectorStoreIndex, top_pairs: int = TOP_PAIRS) -> list[Pair]:
    """For each of my chunks, their most similar chunks; the best pairs first.

    Each of their chunks is used at most once, so one strong chunk can't
    fill every pair.
    """
    only_theirs = MetadataFilters(filters=[MetadataFilter(key="owner", value="them")])
    retriever = index.as_retriever(similarity_top_k=MATCHES_PER_CHUNK, filters=only_theirs)
    mine = [node for node in index.docstore.docs.values() if node.metadata["owner"] == "me"]

    candidates = []
    for node in mine:
        # The stored vector is the query, so nothing new is embedded.
        query = QueryBundle(query_str=node.get_content(),
                            embedding=index.vector_store.get(node.node_id))
        for hit in retriever.retrieve(query):
            candidates.append(Pair(_source(node, 1.0), _source(hit.node, hit.score or 0.0),
                                   hit.score or 0.0))

    pairs, used = [], set()
    for pair in sorted(candidates, key=lambda p: p.score, reverse=True):
        if pair.theirs.text not in used:
            pairs.append(pair)
            used.add(pair.theirs.text)
        if len(pairs) == top_pairs:
            break
    return pairs


def find_common_ground(me: Profile, them: Profile, llm: LLM | None = None,
                       model: str | None = None, embed_model=None) -> CommonGround:
    """Icebreakers about what `me` and `them` share."""
    index = build_profile_index({"me": me, "them": them}, embed_model=embed_model)
    pairs = match_chunks(index)
    text = "\n\n".join(f"Pair {n}:\nA: {p.mine.text}\nB: {p.theirs.text}"
                       for n, p in enumerate(pairs, start=1))
    llm = llm or get_llm(FACTS_GENERATION, model=model)
    reply = llm.predict(COMMON_GROUND_TEMPLATE, me=me.full_name, them=them.full_name,
                        pairs=text)
    return CommonGround(text=reply.strip(), pairs=pairs)
