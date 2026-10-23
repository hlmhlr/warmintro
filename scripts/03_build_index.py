"""Split a profile into chunks, embed them and check retrieval.

Covers the lab's split → index → store steps, and compares the lab's way of
splitting (one line of JSON) with section documents.

Run: uv run python scripts/03_build_index.py
     uv run python scripts/03_build_index.py --profile data/profiles/contacts/lukas_brenner.json
"""

import argparse
import json
import time

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from warmintro.config import CHUNK_SIZE, SIMILARITY_TOP_K
from warmintro.index import (
    create_vector_database,
    profile_to_documents,
    split_profile_data,
    verify_embeddings,
)
from warmintro.llm import describe, get_embeddings
from warmintro.profiles import CONTACTS_DIR, load_profile

QUESTIONS = [
    "Where did this person study?",
    "What does this person do in their free time?",
    "Which programming tools does this person know?",
]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--profile", default=str(CONTACTS_DIR / "amara_okafor.json"))
args = parser.parse_args()
profile = load_profile(args.profile)
print(describe())
print(f"Profile: {profile.full_name}\n")

# 1. The lab's way: the whole profile as one JSON line, default overlap.
lab_text = json.dumps(profile.model_dump(exclude_none=True))
lab_nodes = SentenceSplitter(chunk_size=CHUNK_SIZE).get_nodes_from_documents(
    [Document(text=lab_text)])
print(f"Lab-style chunks: {len(lab_nodes)} (one JSON line of {len(lab_text)} characters)")
if len(lab_nodes) > 1:
    print(f"  chunk 2 starts with: {lab_nodes[1].get_content()[:70]!r}")

# 2. This project: one readable Document per section.
documents = profile_to_documents(profile)
nodes = split_profile_data(documents)
print(f"\nSection chunks: {len(nodes)} from {len(documents)} sections")
for node in nodes:
    print(f"  [{node.metadata['section']:<14}] {node.get_content()[:70]}...")

# 3. Embed. A first tiny call loads the model, so it is timed on its own.
embed_model = get_embeddings()
start = time.perf_counter()
embed_model.get_text_embedding("warm-up")
print(f"\nEmbedding model ready in {time.perf_counter() - start:.1f} s")

start = time.perf_counter()
index = create_vector_database(nodes, embed_model)
print(f"Embedded {len(nodes)} chunks in {time.perf_counter() - start:.1f} s")

report = verify_embeddings(index)
print(f"Embedding check: {'OK' if report.ok else 'PROBLEM'} "
      f"({report.nodes} nodes, dimensions {sorted(report.dimensions)}, "
      f"missing {len(report.missing)}, empty {len(report.empty)})")

# 4. Retrieval only (no LLM yet): which sections come back for a question?
retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
for question in QUESTIONS:
    print(f"\nQ: {question}")
    for hit in retriever.retrieve(question):
        print(f"  {hit.score:.3f}  {hit.node.metadata['section']:<14} "
              f"{hit.node.get_content()[:55]}...")
