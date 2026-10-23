"""Icebreakers about what you and another person share.

Without options it runs the three pairs whose shared details were written
into the sample profiles, counts how many of them the icebreakers name, and
checks in code that each "Shared:" claim appears in both profiles.

Run: uv run python scripts/06_common_ground.py
     uv run python scripts/06_common_ground.py --me alex_rivera --them sofia_marchetti
"""

import argparse
import time

from warmintro.common_ground import check_shared_claims, find_common_ground
from warmintro.config import Generation
from warmintro.evaluate import matches
from warmintro.llm import describe, get_llm
from warmintro.profiles import CONTACTS_DIR, ME_DIR, list_profiles, load_profile

# (me, them, shared topics: each matches if any of its words is mentioned)
PLANTED = [
    ("alex_rivera", "amara_okafor",
     [["Westmere"], ["marathon", "marathons", "running"], ["Dutch"]]),
    ("alex_rivera", "sofia_marchetti",
     [["CoderDojo", "RoboKids", "coding", "children", "teenagers", "workshops"],
      ["computer vision", "vision"]]),
    ("sam_lindqvist", "lukas_brenner",
     [["Altstadt", "conservatory"], ["cello"], ["health", "healthcare", "hospital"],
      ["board games"], ["hiking"]]),
]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--me", choices=list(list_profiles(ME_DIR)))
parser.add_argument("--them", choices=list(list_profiles(CONTACTS_DIR)))
parser.add_argument("--model", help="another Ollama/Gemini model name")
args = parser.parse_args()
runs = [(args.me, args.them, [])] if args.me and args.them else PLANTED

print(describe())
get_llm(Generation(max_new_tokens=5), model=args.model).complete("Say ok")  # load model

found_total = planted_total = claims_ok = claims_total = 0
for me_name, them_name, topics in runs:
    me = load_profile(ME_DIR / f"{me_name}.json")
    them = load_profile(CONTACTS_DIR / f"{them_name}.json")
    start = time.perf_counter()
    result = find_common_ground(me, them, model=args.model)
    seconds = time.perf_counter() - start

    print(f"\n=== {me.full_name} meets {them.full_name} ({seconds:.1f} s)")
    print("Best-matching chunk pairs:")
    for pair in result.pairs:
        print(f"  {pair.score:.3f}  {pair.mine.section:<13} ↔ {pair.theirs.section:<13} "
              f"words in both: {', '.join(pair.shared) or 'none'}")
    print(f"\n{result.text}")

    checks = check_shared_claims(result.text, me, them)
    for check in checks:
        if not check.supported:
            print(f"  ! Not shared: {check.claim!r} ({', '.join(check.one_sided)} "
                  "is in only one profile)")
    claims_ok += sum(c.supported for c in checks)
    claims_total += len(checks)
    if topics:
        found = [t[0] for t in topics if matches(result.text, [t])]
        print(f"\nPlanted shared topics mentioned: {len(found)}/{len(topics)} "
              f"({', '.join(found) or 'none'})")
        found_total += len(found)
        planted_total += len(topics)

print()
if planted_total:
    print(f"Planted shared topics mentioned: {found_total}/{planted_total}")
print(f"Shared claims found in both profiles: {claims_ok}/{claims_total}")
