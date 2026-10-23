"""Check that the chosen models are installed and answer.

Run: uv run python scripts/00_check_setup.py
"""

import sys
import time

from warmintro.config import Generation, load_settings
from warmintro.llm import describe, get_embeddings, get_llm

settings = load_settings()
print(describe(settings))

# Ollama only: are the models downloaded?
uses_ollama = "ollama" in (settings.llm_provider, settings.embeddings_provider)
if uses_ollama:
    import ollama

    try:
        installed = {m.model for m in ollama.Client(settings.ollama_base_url).list().models}
    except ConnectionError:
        sys.exit(f"\nOllama is not running at {settings.ollama_base_url}. "
                 "Start the Ollama app, then run this again.")
    needed = []
    if settings.llm_provider == "ollama":
        needed.append(settings.ollama_model)
    if settings.embeddings_provider == "ollama":
        needed.append(settings.ollama_embed_model)
    missing = [name for name in needed if name not in installed]
    if missing:
        print("\nMissing Ollama models. Download them with:")
        for name in missing:
            print(f"  ollama pull {name}")
        sys.exit(1)
    print("Ollama models installed:", ", ".join(needed))

# One short answer. The first call to Ollama also loads the model into memory,
# so it is timed separately from a second, steady-state call.
llm = get_llm(Generation(max_new_tokens=10), settings=settings)
for label in ("first call (includes loading)", "second call"):
    start = time.perf_counter()
    answer = llm.complete("Reply with one word: ready").text.strip()
    print(f"LLM {label}: {answer!r} in {time.perf_counter() - start:.1f} s")

start = time.perf_counter()
vector = get_embeddings(settings).get_text_embedding("Hello from warmintro")
print(f"Embedding: {len(vector)} numbers in {time.perf_counter() - start:.1f} s")

print("\nAll set.")
