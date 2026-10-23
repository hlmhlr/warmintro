"""Scorecard: how grounded are the answers and facts? (the RAG evaluation stage)

Run: uv run python scripts/07_evaluate.py --label fixed
     uv run python scripts/07_evaluate.py --label quick --runs 1 --profiles lukas_brenner
     uv run python scripts/07_evaluate.py --rescore baseline        # re-judge a saved run
     uv run python scripts/07_evaluate.py --label fixed --judge gemini
     uv run python scripts/07_evaluate.py --calibrate               # test the judge itself
Writes outputs/scorecard-<label>.md and outputs/results-<label>.json.
About 5 minutes on a GTX 1660 Ti with the defaults (3 profiles, 3 facts runs).
"""

import argparse
import dataclasses
import time

from warmintro.config import ANSWER_GENERATION, Generation, load_settings
from warmintro.errors import friendly_error
from warmintro.evaluate import (
    calibrate_judge,
    evaluate_facts,
    evaluate_questions,
    load_questions,
    load_results,
    rejudge,
    save_results,
    scorecard,
)
from warmintro.index import build_profile_index
from warmintro.llm import describe, get_llm
from warmintro.profiles import CONTACTS_DIR, ROOT, load_profile

OUT = ROOT / "outputs"

questions = load_questions()
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--label", default="run", help="name for this run, e.g. fixed")
parser.add_argument("--runs", type=int, default=3, help="facts generations per profile")
parser.add_argument("--profiles", nargs="+", default=list(questions), choices=list(questions))
parser.add_argument("--judge", choices=["local", "gemini", "none"], default="local",
                    help="model that judges the facts (gemini needs: uv sync --extra gemini)")
parser.add_argument("--rescore", metavar="LABEL",
                    help="only judge the facts of a saved run again")
parser.add_argument("--calibrate", action="store_true",
                    help="only test the judge on statements with known answers")
args = parser.parse_args()

settings = load_settings()
judge_llm = None
try:
    if args.judge != "none":
        judge_settings = dataclasses.replace(settings, llm_provider="ollama"
                                             if args.judge == "local" else "gemini")
        judge_llm = get_llm(ANSWER_GENERATION, settings=judge_settings)
except Exception as error:
    raise SystemExit(friendly_error(error)) from None
judge_name = getattr(judge_llm, "model", "none")
model_line = f"{describe(settings)} | judge: {judge_name}"
print(model_line)

started = time.perf_counter()
if args.calibrate:
    if judge_llm is None:
        raise SystemExit("--calibrate needs a judge (local or gemini).")
    names = ("amara_okafor", "lukas_brenner", "sofia_marchetti")
    profiles = {name: load_profile(CONTACTS_DIR / f"{name}.json") for name in names}
    c = calibrate_judge(judge_llm, profiles)
    print(f"\nJudge calibration ({judge_name})")
    print(f"  True statements judged YES:   {c.true_accepted}/{c.true_total}")
    print(f"  Planted errors judged NO:     {c.false_rejected}/{c.false_total}")
    print(f"  Unclear replies:              {c.unclear}")
    print(f"  Accuracy: {c.accuracy:.0%} (a judge that always says YES scores 50%)")
    for miss in c.misses:
        print(f"  wrong -> {miss}")
    raise SystemExit(0)
if args.rescore:
    # Same stored answers, current judge: makes two runs comparable.
    results = load_results(OUT / f"results-{args.rescore}.json")
    names = {r.profile for r in results}
    profiles = {name: load_profile(CONTACTS_DIR / f"{name}.json") for name in names}
    rejudge(results, profiles, judge_llm)
    label = f"{args.rescore}-rejudged-{args.judge}"
else:
    get_llm(Generation(max_new_tokens=5)).complete("Say ok")  # load the model first
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
    label = args.label if args.judge == "local" else f"{args.label}-{args.judge}"

report = scorecard(results, label, model_line)
OUT.mkdir(exist_ok=True)
(OUT / f"scorecard-{label}.md").write_text(report, encoding="utf-8")
save_results(results, OUT / f"results-{label}.json")

print("\n" + report.split("\n## Failures")[0])
print(f"Total time {(time.perf_counter() - started) / 60:.1f} min. "
      f"Details: outputs/scorecard-{label}.md")
