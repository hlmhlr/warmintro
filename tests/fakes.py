"""Fake models for tests: no Ollama, no network."""

import re
import zlib

from llama_index.core.base.embeddings.base import BaseEmbedding


STOP_WORDS = set(
    "a an and at did does do for from has have he her his i in is it of on or "
    "she the their they this to was what where which who with".split()
)


class KeywordEmbedding(BaseEmbedding):
    """Counts words into a vector, so texts sharing words are similar.

    Good enough to test retrieval: "Where did she study?" lands near the
    chunk that contains "studied". Every text it sees is recorded.
    """

    dimensions: int = 1024
    seen: list[str] = []

    def _vector(self, text: str) -> list[float]:
        self.seen.append(text)
        vector = [0.0] * self.dimensions
        for word in re.findall(r"[a-z]+", text.lower()):
            if word in STOP_WORDS:
                continue
            word = word.removesuffix("ied").removesuffix("ies").removesuffix("y")
            vector[zlib.crc32(word.encode()) % self.dimensions] += 1.0
        return vector

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)
