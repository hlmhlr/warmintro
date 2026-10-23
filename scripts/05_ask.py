"""Ask fixed questions about a profile, including some it can't answer.

The last two questions have no answer in the profile, so a grounded model
should reply "I don't know."

Run: uv run python scripts/05_ask.py
     uv run python scripts/05_ask.py --profile data/profiles/contacts/lukas_brenner.json
"""

import argparse
import time

from warmintro.config import Generation
from warmintro.index import build_profile_index
from warmintro.llm import describe, get_llm
from warmintro.profiles import CONTACTS_DIR, load_profile
from warmintro.query import answer_user_query

QUESTIONS = [
    "What is this person's current job title?",
    "Where did this person study?",
    "Which languages does this person speak?",
    "What does this person do outside of work?",
    "What is this person's salary?",          # not in the profile
    "How many children does this person have?",  # not in the profile
]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--profile", default=str(CONTACTS_DIR / "amara_okafor.json"))
parser.add_argument("--model", help="another Ollama/Gemini model name")
args = parser.parse_args()

print(describe())
profile = load_profile(args.profile)
index = build_profile_index({"them": profile})
get_llm(Generation(max_new_tokens=5), model=args.model).complete("Say ok")  # load model
print(f"Questions about {profile.full_name}\n")

times = []
for question in QUESTIONS:
    start = time.perf_counter()
    answer = answer_user_query(index, question, model=args.model)
    times.append(time.perf_counter() - start)
    print(f"Q: {question}\nA: {answer.text}")
    print(f"   ({times[-1]:.1f} s, sections: {', '.join(answer.sections)})\n")

print(f"Average {sum(times) / len(times):.1f} s per answer")
