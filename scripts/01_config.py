"""Show the configuration (replaces the lab's test_config.py).

Run: uv run python scripts/01_config.py
"""

from warmintro import config
from warmintro.llm import describe

print("Initial facts template defined:", bool(config.INITIAL_FACTS_TEMPLATE))
print("User question template defined:", bool(config.USER_QUESTION_TEMPLATE))
print("Chunk size (tokens):", config.CHUNK_SIZE)
print("Similarity top k:", config.SIMILARITY_TOP_K)
print("Facts generation:", config.FACTS_GENERATION.options())
print("Answer generation:", config.ANSWER_GENERATION.options())
print(describe())

print("\n--- Initial facts template ---")
print(config.INITIAL_FACTS_TEMPLATE)
print("\n--- User question template ---")
print(config.USER_QUESTION_TEMPLATE)
