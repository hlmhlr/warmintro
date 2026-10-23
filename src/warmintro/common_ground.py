"""Find what you and another person share, and turn it into icebreakers.

Both profiles go into one index, tagged owner="me" or owner="them". Each of
your chunks is used as a query against their chunks only, through a metadata
filter, reusing the vectors already stored (no extra embedding calls).

Similarity alone favoured pairs of the same kind ("Skills of A" vs "Skills
of B" scored about 0.78 whatever the skills), so a really shared school lost.
Pairs that share actual words (a school, a language, an instrument) now come
first, and those words are given to the LLM. The reply is cut to three
icebreakers, and check_shared_claims() tests each "Shared:" line in code.
"""

import re

from dataclasses import dataclass, field

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from warmintro.config import FACTS_GENERATION
from warmintro.index import build_profile_index, profile_to_texts
from warmintro.llm import get_llm
from warmintro.profiles import Profile
from warmintro.query import Source

TOP_PAIRS = 5        # pairs sent to the LLM
MATCHES_PER_CHUNK = 2  # people often have two schools or several jobs
MAX_ICEBREAKERS = 3

# Words that two chunks share without meaning a real common interest: the
# sentence patterns of the section texts, and very general words.
STOP_WORDS = set("""
and are but can for from has have her his how its not that the their them
then they this was what when where which who why will with you your
""".split())
IGNORED_WORDS = set("""
about also and are been both cause current education experience from have
interests into job jobs language languages more past present project projects
skills speaks studied studies their them they this volunteers with work worked
works year years university school degree company based other build builds
built helps helped runs makes made used uses
""".split())

COMMON_GROUND_TEMPLATE = PromptTemplate("""You help {me} start a conversation with {them} at a networking event.
Below are pairs of profile excerpts that may have something in common.
In each pair, line A is about {me} and line B is about {them}.

{pairs}

Write up to three icebreakers about things both people really share, such as
the same school, city, skill, hobby, cause or career change.

Rules:
- Start your reply with "1." Write no introduction and no closing line.
- Something is shared only if it appears in both line A and line B. The
  "Words in both" line of each pair shows candidates. Skip a pair that shares
  nothing, and never list something only one person has.
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
    shared: list[str] = field(default_factory=list)  # words in both chunks


@dataclass
class CommonGround:
    text: str
    pairs: list[Pair]


def _word_set(text: str, ignore: set[str]) -> dict[str, str]:
    """Meaningful words of a text: {normalized: as written}."""
    words = {}
    for word in re.findall(r"[A-Za-z][A-Za-z+#-]{2,}", text):
        key = word.lower().rstrip("s") if len(word) > 4 else word.lower()
        if key not in ignore and word.lower() not in ignore and len(key) >= 3:
            words.setdefault(key, word)
    return words


def _name_words(*names: str) -> set[str]:
    return {part.lower() for name in names for part in name.split()}


def shared_terms(text_a: str, text_b: str, names: set[str] = frozenset()) -> list[str]:
    """Words in both texts, ignoring names, sentence patterns and stop words.

    "marathons" and "Marathon" count as the same word.
    """
    ignore = IGNORED_WORDS | set(names) | STOP_WORDS
    a, b = _word_set(text_a, ignore), _word_set(text_b, ignore)
    return [a[key] for key in a if key in b]


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

    names = _name_words(*{node.metadata["person"] for node in index.docstore.docs.values()})
    candidates = []
    for node in mine:
        # The stored vector is the query, so nothing new is embedded.
        query = QueryBundle(query_str=node.get_content(),
                            embedding=index.vector_store.get(node.node_id))
        for hit in retriever.retrieve(query):
            candidates.append(Pair(
                _source(node, 1.0), _source(hit.node, hit.score or 0.0), hit.score or 0.0,
                shared_terms(node.get_content(), hit.node.get_content(), names)))

    return rank_pairs(candidates, top_pairs)


def rank_pairs(candidates: list[Pair], top_pairs: int = TOP_PAIRS) -> list[Pair]:
    """Pairs with shared words first, then by similarity; each of their chunks once."""
    pairs, used = [], set()
    for pair in sorted(candidates, key=lambda p: (bool(p.shared), p.score), reverse=True):
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
    text = "\n\n".join(
        f"Pair {n}:\nA: {p.mine.text}\nB: {p.theirs.text}\n"
        f"Words in both: {', '.join(p.shared) or 'none'}"
        for n, p in enumerate(pairs, start=1))
    llm = llm or get_llm(FACTS_GENERATION, model=model)
    reply = llm.predict(COMMON_GROUND_TEMPLATE, me=me.full_name, them=them.full_name,
                        pairs=text)
    return CommonGround(text=limit_items(reply.strip(), MAX_ICEBREAKERS), pairs=pairs)


def limit_items(text: str, limit: int) -> str:
    """Keep only the first `limit` numbered items (the model sometimes adds more)."""
    match = re.search(rf"(?m)^\s*{limit + 1}[.)]\s", text)
    return text[:match.start()].rstrip() if match else text


@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    one_sided: list[str]  # words found in only one of the two profiles


def check_shared_claims(text: str, me: Profile, them: Profile) -> list[ClaimCheck]:
    """Check each "Shared: ..." line against both profiles, in code.

    A word of the claim that appears in one profile but not the other means
    the claim isn't really shared ("German, Swedish, English" when only one
    person speaks Swedish). Words in neither profile are the model's own
    wording ("Python programming") and are ignored. Being word-based, the
    check can't catch a wrong combination of words that are in both.
    """
    names = _name_words(me.full_name, them.full_name)
    ignore = IGNORED_WORDS | names | STOP_WORDS
    mine = _word_set(" ".join(t for _, t in profile_to_texts(me)), ignore)
    theirs = _word_set(" ".join(t for _, t in profile_to_texts(them)), ignore)
    checks = []
    for claim in re.findall(r"(?im)^\s*\d+[.)]\s*shared:\s*(.+)$", text):
        words = _word_set(claim, ignore)
        one_sided = [w for key, w in words.items() if (key in mine) != (key in theirs)]
        checks.append(ClaimCheck(claim.strip(), not one_sided, one_sided))
    return checks
