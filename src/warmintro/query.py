"""Generate conversation starters and answer questions about a profile.

Replaces the lab's query_engine.py. Two fixes:
- The lab's answer_user_query() retrieved the chunks, built a context string,
  threw it away, then let the query engine retrieve again: every question was
  embedded and searched twice. Here each question is retrieved once.
- On failure the lab returned a plain string, but callers read `.response`
  from it and crashed. Here errors are raised and callers decide what to show.
"""

from dataclasses import dataclass

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.llms import LLM

from warmintro.config import (
    ANSWER_GENERATION,
    FACTS_GENERATION,
    FACTS_QUERY,
    FACTS_TOP_K,
    INITIAL_FACTS_TEMPLATE,
    SIMILARITY_TOP_K,
    USER_QUESTION_TEMPLATE,
)
from warmintro.llm import get_llm


@dataclass
class Source:
    """A retrieved chunk the answer was based on."""

    person: str
    section: str
    score: float
    text: str


@dataclass
class Answer:
    text: str
    sources: list[Source]

    @property
    def sections(self) -> list[str]:
        """Section names in retrieval order, without repeats."""
        return list(dict.fromkeys(source.section for source in self.sources))


def _ask(index: VectorStoreIndex, query: str, template: str, top_k: int, llm: LLM) -> Answer:
    """Retrieve once, fill the template, ask the LLM."""
    engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
        text_qa_template=PromptTemplate(template),
        response_mode="compact",  # all chunks in one prompt, one LLM call
    )
    response = engine.query(query)
    sources = [
        Source(
            person=hit.node.metadata.get("person", ""),
            section=hit.node.metadata.get("section", ""),
            score=hit.score or 0.0,
            text=hit.node.get_content(),
        )
        for hit in response.source_nodes
    ]
    return Answer(text=(response.response or "").strip(), sources=sources)


def generate_initial_facts(
    index: VectorStoreIndex, llm: LLM | None = None, model: str | None = None
) -> Answer:
    """Three interesting facts, each with a conversation-starter question."""
    llm = llm or get_llm(FACTS_GENERATION, model=model)
    return _ask(index, FACTS_QUERY, INITIAL_FACTS_TEMPLATE, FACTS_TOP_K, llm)


def answer_user_query(
    index: VectorStoreIndex, question: str, llm: LLM | None = None, model: str | None = None
) -> Answer:
    """Answer a question from the profile only ("I don't know." otherwise)."""
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")
    llm = llm or get_llm(ANSWER_GENERATION, model=model)
    return _ask(index, question, USER_QUESTION_TEMPLATE, SIMILARITY_TOP_K, llm)
