"""Check each free cloud provider separately: key, model and speed.

Never prints a key. Needs: uv sync --extra cloud
Run: uv run python scripts/00_check_cloud.py
"""

import dataclasses
import time

from warmintro.config import CLOUD_CHAIN, Generation, load_settings
from warmintro.errors import friendly_error
from warmintro.llm import describe_llm, get_embeddings, get_llm


def reason(error: Exception) -> str:
    """Friendly message plus the error type and its first words, for diagnosis."""
    detail = " ".join(str(error).split())[:120]
    return f"{friendly_error(error)}\n          ({type(error).__name__}: {detail})"


settings = load_settings()
generation = Generation(max_new_tokens=10)
working = []

print("LLM providers (one short answer each):")
for provider in CLOUD_CHAIN.split(","):
    single = dataclasses.replace(settings, llm_provider=provider)
    label = describe_llm(single)
    start = time.perf_counter()
    try:
        answer = get_llm(generation, settings=single).complete("Reply with one word: ready")
        print(f"  OK    {label}: {answer.text.strip()[:40]!r} in "
              f"{time.perf_counter() - start:.1f} s")
        working.append(provider)
    except Exception as error:
        print(f"  FAIL  {label}: {reason(error)}")

print("\nEmbeddings (Cloudflare):")
start = time.perf_counter()
try:
    single = dataclasses.replace(settings, embeddings_provider="cloudflare")
    vector = get_embeddings(single).get_text_embedding("Hello from warmintro")
    print(f"  OK    {len(vector)} numbers in {time.perf_counter() - start:.1f} s")
except Exception as error:
    print(f"  FAIL  {reason(error)}")

print(f"\n{len(working)} of 3 LLM providers work: {', '.join(working) or 'none'}")
