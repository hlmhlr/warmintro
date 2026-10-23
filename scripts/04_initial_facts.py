"""Generate three facts with conversation starters (the lab's initial facts).

Run: uv run python scripts/04_initial_facts.py
     uv run python scripts/04_initial_facts.py --profile data/profiles/contacts/sofia_marchetti.json
"""

import argparse
import time

from warmintro.config import FACTS_GENERATION, Generation
from warmintro.index import build_profile_index
from warmintro.llm import describe, get_llm
from warmintro.profiles import CONTACTS_DIR, load_profile
from warmintro.query import generate_initial_facts

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--profile", default=str(CONTACTS_DIR / "amara_okafor.json"))
parser.add_argument("--model", help="another Ollama/Gemini model name")
args = parser.parse_args()

print(describe())
profile = load_profile(args.profile)
index = build_profile_index({"them": profile})
print(f"Indexed {profile.full_name}")

# Load the model first, so the timing below is the steady-state speed.
start = time.perf_counter()
get_llm(Generation(max_new_tokens=5), model=args.model).complete("Say ok")
print(f"Model loaded in {time.perf_counter() - start:.1f} s")
print(f"Settings: {FACTS_GENERATION.options()}\n")

start = time.perf_counter()
facts = generate_initial_facts(index, model=args.model)
seconds = time.perf_counter() - start

print(f"Here are 3 interesting facts about {profile.full_name}:\n")
print(facts.text)
print(f"\n({seconds:.1f} s, from sections: {', '.join(facts.sections)})")
