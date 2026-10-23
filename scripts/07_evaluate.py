"""Scorecard: how grounded are the answers and facts? (the RAG evaluation stage)

Run: uv run python scripts/07_evaluate.py --label baseline
     uv run python scripts/07_evaluate.py --label quick --runs 1 --profiles lukas_brenner
Writes outputs/scorecard-<label>.md and outputs/results-<label>.json.
About 5 minutes on a GTX 1660 Ti with the defaults (3 profiles, 3 facts runs).
"""

import argparse
import time

from warmintro.config import ANSWER_GENERATION, Generation
from warmintro.evaluate import (
    evaluate_facts,
    evaluate_questions,
    load_questions,
    save_results,
    scorecard,
)
from warmintro.index import build_profile_index
from warmintro.llm import describe, get_llm
from warmintro.profiles import CONTACTS_DIR, ROOT, load_profile

questions = load_questions()
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--label", default="run", help="name for this run, e.g. baseline")
parser.add_argument("--runs", type=int, default=3, help="facts generations per profile")
parser.add_argument("--profiles", nargs="+", default=list(questions),
                    choices=list(questions))
parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
args = parser.parse_args()

model_line = describe()
print(model_line)
get_llm(Generation(max_new_tokens=5)).complete("Say ok")  # load the model first
judge_llm = None if args.no_judge else get_llm(ANSWER_GENERATION)

started = time.perf_counter()
results = []
for name in args.profiles:
    profile = load_profile(CONTACTS_DIR / f"{name}.json")
    index = build_profile_index({"them": profile})
    print(f"\n{profile.full_name}")
    new = evaluate_questions(name, index, questions[name])
    new += evaluate_facts(name, index, profile, args.runs, judge_llm=judge_llm)
    for r in new:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.kind:<12} {r.question}"
              f"{'  -> ' + r.detail if r.detail else ''}")
    results += new

report = scorecard(results, args.label, model_line)
out = ROOT / "outputs"
(out / f"scorecard-{args.label}.md").parent.mkdir(exist_ok=True)
(out / f"scorecard-{args.label}.md").write_text(report, encoding="utf-8")
save_results(results, out / f"results-{args.label}.json")

print("\n" + report.split("\n## Failures")[0])
print(f"Total time {(time.perf_counter() - started) / 60:.1f} min. "
      f"Details: outputs/scorecard-{args.label}.md")
