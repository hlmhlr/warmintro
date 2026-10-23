"""Command-line icebreaker bot (replaces the lab's main.py).

Run: uv run warmintro                         # pick a sample profile
     uv run warmintro --profile lukas_brenner
     uv run warmintro --profile path/to/profile.json --model granite4:micro-h
     uv run warmintro --profile amara_okafor --me alex_rivera   # + common ground

The lab's --url, --api-key and --mock options are gone: profiles are local
files, so there is nothing to fetch.
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from llama_index.core import VectorStoreIndex

from warmintro.common_ground import find_common_ground
from warmintro.errors import friendly_error
from warmintro.index import build_profile_index
from warmintro.llm import describe
from warmintro.profiles import CONTACTS_DIR, ME_DIR, list_profiles, load_profile
from warmintro.query import answer_user_query, generate_initial_facts

EXIT_WORDS = {"exit", "quit", "bye"}


def resolve_profile(value: str, folder: Path = CONTACTS_DIR) -> Path:
    """Accept a file path or the name of a sample profile."""
    path = Path(value)
    if path.is_file():
        return path
    samples = list_profiles(folder)
    if value in samples:
        return samples[value]
    raise SystemExit(f"No profile file or sample named {value!r}. "
                     f"Samples: {', '.join(samples)}")


def choose_profile(input_fn: Callable[[str], str] | None = None,
                   folder: Path = CONTACTS_DIR) -> Path:
    """Let the user pick a sample profile by number (Enter = the first)."""
    input_fn = input_fn or input
    samples = list(list_profiles(folder).items())
    print("Sample profiles:")
    for number, (name, _) in enumerate(samples, start=1):
        print(f"  {number}. {name}")
    choice = input_fn("Choose a number (Enter for 1): ").strip() or "1"
    if not choice.isdigit() or not 1 <= int(choice) <= len(samples):
        raise SystemExit(f"Please choose a number from 1 to {len(samples)}.")
    return samples[int(choice) - 1][1]


def chat(index: VectorStoreIndex, model: str | None = None,
         input_fn: Callable[[str], str] | None = None) -> None:
    """Answer questions until the user types exit, quit or bye."""
    input_fn = input_fn or input
    print("\nAsk anything about this person. Type 'exit', 'quit' or 'bye' to stop.")
    while True:
        try:
            question = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            question = "exit"
        if question.lower() in EXIT_WORDS:
            print("Bot: Goodbye!")
            return
        if not question:
            continue
        # The lab slept 1 second here to fake typing; real answers need no delay.
        try:
            start = time.perf_counter()
            answer = answer_user_query(index, question, model=model)
            print(f"Bot: {answer.text}  ({time.perf_counter() - start:.1f} s)\n")
        except Exception as error:  # keep the chat going after a failed answer
            print(f"Bot: {friendly_error(error)}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Icebreakers from a professional profile")
    parser.add_argument("--profile", help="profile JSON file or sample name")
    parser.add_argument("--me", help="your own profile (file or sample name), "
                                     "to also show what you have in common")
    parser.add_argument("--model", help="another Ollama/Gemini model name")
    parser.add_argument("--verbose", action="store_true", help="show error details")
    args = parser.parse_args(argv)
    # Normally the user sees only the friendly message; --verbose adds the log.
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.CRITICAL,
                        stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")

    path = resolve_profile(args.profile) if args.profile else choose_profile()
    me_path = resolve_profile(args.me, ME_DIR) if args.me else None
    print(describe())
    try:
        profile = load_profile(path)
        index = build_profile_index({"them": profile})
        facts = generate_initial_facts(index, model=args.model)
        shared = (find_common_ground(load_profile(me_path), profile, model=args.model)
                  if me_path else None)
    except Exception as error:
        raise SystemExit(friendly_error(error)) from None

    print(f"\nHere are 3 interesting facts about {profile.full_name}:\n")
    print(facts.text)
    if shared:
        print(f"\nWhat you have in common with {profile.full_name}:\n")
        print(shared.text)
    chat(index, model=args.model)


if __name__ == "__main__":
    main()
