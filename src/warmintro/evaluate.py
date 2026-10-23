"""Measure how grounded the answers and facts are: the RAG "evaluation" stage.

The lab lists evaluation as the fifth RAG stage but skips it. This module
checks, with a fixed question set:
- answerable questions: the answer mentions the expected words,
- unanswerable questions: the answer is "I don't know.",
- qualifier questions: the answer keeps details such as "basic" Italian,
- facts: the format (3 numbered facts with questions), numbers that don't
  appear in the profile, whether a job was used, and an LLM judge.

Most checks are plain string rules, so they are fast and repeatable. Only
the judge uses a model.
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.llms import LLM

from warmintro.index import profile_to_texts
from warmintro.profiles import ROOT, Profile
from warmintro.query import answer_user_query, generate_initial_facts

QUESTIONS_PATH = ROOT / "data" / "eval" / "questions.json"

# LlamaIndex's FaithfulnessEvaluator passes an answer if "yes" appears
# anywhere in the judge's reply. This judge must start with YES or NO, and
# gets one fact at a time: in the baseline, a small model judging all three
# facts at once flagged questions and quoted the profile as "unsupported".
JUDGE_TEMPLATE = PromptTemplate("""You check one statement about a person against their profile.

Profile:
---------------------
{profile}
---------------------
Statement:
{statement}

Is the statement supported by the profile? A statement that summarises or
rephrases the profile counts as supported. Details that the profile does not
give (dates, numbers, places, reasons, opinions) are not supported.
Reply on the first line with exactly YES or NO.
If NO, on the second line copy the unsupported words from the statement.""")

def load_questions(path: Path = QUESTIONS_PATH) -> dict:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return {name: value for name, value in data.items() if not name.startswith("_")}


def _contains(text: str, word: str) -> bool:
    """Whole-word, case-insensitive match ("no" doesn't match "know")."""
    pattern = re.escape(word.lower())
    if word[0].isalnum():
        pattern = r"\b" + pattern
    if word[-1].isalnum():
        pattern += r"\b"
    return re.search(pattern, text.lower()) is not None


def matches(text: str, expect: list) -> bool:
    """Every item must appear; an item that is a list needs any one of its words."""
    return all(
        any(_contains(text, word) for word in (item if isinstance(item, list) else [item]))
        for item in expect
    )


def says_dont_know(text: str) -> bool:
    normalized = text.lower().replace("’", "'")
    return "i don't know" in normalized or "i do not know" in normalized


def check_facts_format(text: str) -> str | None:
    """None if the text is 3 numbered facts, each with a question; else the problem."""
    items = re.split(r"(?m)^\s*\d+[.)]\s+", text)
    numbers = re.findall(r"(?m)^\s*(\d+)[.)]\s+", text)
    if numbers != ["1", "2", "3"]:
        return f"expected facts numbered 1-3, got {numbers or 'none'}"
    if items[0].strip():
        return "text before fact 1"
    if not all("?" in item for item in items[1:]):
        return "a fact has no question"
    return None


def unsupported_numbers(text: str, profile_text: str) -> list[str]:
    """Numbers in the text that appear nowhere in the profile (e.g. an invented year).

    It can't catch a number that exists elsewhere in the profile but is
    used wrongly, so the judge below is still needed.
    """
    without_list_numbers = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    found = re.findall(r"\d+(?:[.,]\d+)*", without_list_numbers)
    return [number for number in dict.fromkeys(found) if number not in profile_text]


def split_facts(text: str) -> list[str]:
    """The numbered facts, or the whole text if it isn't numbered."""
    items = [item.strip() for item in re.split(r"(?m)^\s*\d+[.)]\s+", text)[1:]]
    return [item for item in items if item] or [text.strip()]


def statement_only(fact: str) -> str:
    """Remove the conversation-starter question, so only claims are judged."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", fact)
    kept = [s for s in sentences if s.strip() and not s.strip().rstrip('"\'').endswith("?")]
    text = " ".join(kept)
    # Labels such as "Question:" or "Conversation starter:" left without a question.
    text = re.sub(r"[-*\s]*(question|conversation starter|friendly question)\s*:\s*$", "",
                  text, flags=re.IGNORECASE)
    return text.strip(" -*")


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}


def judge_statement(statement: str, profile_text: str, llm: LLM) -> tuple[bool | None, str]:
    """(True / False / None if the judge's reply can't be trusted, reason)."""
    reply = llm.predict(JUDGE_TEMPLATE, profile=profile_text, statement=statement).strip()
    first_word = re.sub(r"[^A-Z]", "", reply.split()[0].upper()) if reply else ""
    quote = " ".join(reply.splitlines()[1:]).strip()
    if first_word == "YES":
        return True, ""
    if first_word != "NO":
        return None, f"unclear judge reply: {reply[:80]!r}"
    if quote:
        # A NO must point at words from the statement. In the baseline the
        # judge often quoted the profile instead, which says nothing.
        quoted = _words(quote)
        if quoted and len(quoted & _words(statement)) / len(quoted) < 0.5:
            return None, f"judge quoted text that isn't in the statement: {quote[:80]!r}"
    return False, quote or "no reason given"


def judge_grounding(text: str, profile_text: str, llm: LLM) -> tuple[bool | None, str]:
    """Judge each fact separately. False if any fact is unsupported,
    None if no fact failed but some judge replies were unclear."""
    unclear = []
    for number, fact in enumerate(split_facts(text), start=1):
        statement = statement_only(fact)
        if not statement:
            continue
        grounded, reason = judge_statement(statement, profile_text, llm)
        if grounded is False:
            return False, f"fact {number}: {reason}"
        if grounded is None:
            unclear.append(f"fact {number}: {reason}")
    if unclear:
        return None, "; ".join(unclear)
    return True, ""


@dataclass
class Result:
    profile: str
    kind: str          # answerable | unanswerable | qualifier | facts
    question: str
    answer: str
    passed: bool
    detail: str = ""
    seconds: float = 0.0
    sections: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)  # facts only: each check's result


def evaluate_questions(name: str, index: VectorStoreIndex, questions: dict,
                       llm: LLM | None = None) -> list[Result]:
    results = []
    for kind in ("answerable", "unanswerable", "qualifier"):
        items = questions.get(kind + "s" if kind == "qualifier" else kind, [])
        for item in items:
            question = item if isinstance(item, str) else item["question"]
            start = time.perf_counter()
            answer = answer_user_query(index, question, llm=llm)
            seconds = time.perf_counter() - start
            if kind == "unanswerable":
                passed = says_dont_know(answer.text)
                detail = "" if passed else "answered instead of 'I don't know'"
            elif says_dont_know(answer.text):
                passed, detail = False, "said 'I don't know'"
            else:
                passed = matches(answer.text, item["expect"])
                detail = "" if passed else f"missing {item['expect']}"
            results.append(Result(name, kind, question, answer.text, passed, detail,
                                  round(seconds, 2), answer.sections))
    return results


def profile_text_of(profile: Profile) -> str:
    """The whole profile as the judge sees it."""
    return "\n".join(text for _, text in profile_to_texts(profile))


def evaluate_facts(name: str, index: VectorStoreIndex, profile: Profile, runs: int,
                   llm: LLM | None = None, judge_llm: LLM | None = None) -> list[Result]:
    """Generate the facts `runs` times (they are sampled, so they vary)."""
    profile_text = profile_text_of(profile)
    results = []
    for run in range(1, runs + 1):
        start = time.perf_counter()
        facts = generate_initial_facts(index, llm=llm)
        seconds = time.perf_counter() - start
        problems = []

        format_problem = check_facts_format(facts.text)
        numbers = unsupported_numbers(facts.text, profile_text)
        used_job = "experience" in facts.sections
        checks = {"format": format_problem is None, "numbers": not numbers,
                  "used_job": used_job}
        if format_problem:
            problems.append(format_problem)
        if numbers:
            problems.append(f"numbers not in profile: {', '.join(numbers)}")
        if not used_job:
            problems.append("no job in the retrieved chunks")
        if judge_llm is not None:
            grounded, reason = judge_grounding(facts.text, profile_text, judge_llm)
            checks["judge"] = grounded
            if grounded is False:
                problems.append(f"judge: {reason}")
            elif grounded is None:  # not counted as a failure of the facts
                checks["judge_note"] = reason

        results.append(Result(name, "facts", f"facts run {run}", facts.text,
                              not problems, "; ".join(problems), round(seconds, 2),
                              facts.sections, checks))
    return results


def rejudge(results: list[Result], profiles: dict[str, Profile], judge_llm: LLM) -> None:
    """Judge stored facts again with the current judge (no new generation).

    Lets an old run be compared with a new one using the same judge.
    """
    for r in results:
        if r.kind != "facts":
            continue
        grounded, reason = judge_grounding(r.answer, profile_text_of(profiles[r.profile]),
                                           judge_llm)
        r.checks["judge"] = grounded
        r.checks.pop("judge_note", None)
        problems = [p for p in r.detail.split("; ") if p and not p.startswith("judge:")]
        if grounded is False:
            problems.append(f"judge: {reason}")
        elif grounded is None:
            r.checks["judge_note"] = reason
        r.detail = "; ".join(problems)
        r.passed = not problems


def load_results(path: Path) -> list[Result]:
    with open(path, encoding="utf-8") as file:
        return [Result(**item) for item in json.load(file)]


def _count(results: list[Result], kind: str, check: str | None = None) -> str:
    items = [r for r in results if r.kind == kind]
    if not items:
        return "-"
    if check is None:
        passed = sum(r.passed for r in items)
    elif check == "judge_unclear":
        if "judge" not in items[0].checks:
            return "-"
        passed = sum(r.checks["judge"] is None for r in items)
    else:
        if check not in items[0].checks:
            return "-"
        passed = sum(r.checks[check] is True for r in items)
    return f"{passed}/{len(items)}"


def _average(results: list[Result], kinds: tuple[str, ...]) -> str:
    times = [r.seconds for r in results if r.kind in kinds]
    return f"{sum(times) / len(times):.1f} s" if times else "-"


ROWS = [
    ("Answerable questions correct", "answerable", None),
    ("Unanswerable → \"I don't know.\"", "unanswerable", None),
    ("Qualifier kept (e.g. \"basic\")", "qualifier", None),
    ("Facts: 3 numbered, each with a question", "facts", "format"),
    ("Facts: no numbers missing from the profile", "facts", "numbers"),
    ("Facts: a job was in the context", "facts", "used_job"),
    ("Facts: judge says grounded", "facts", "judge"),
    ("Facts: judge reply unclear (not counted)", "facts", "judge_unclear"),
    ("Facts: passed every check", "facts", None),
]


def scorecard(results: list[Result], title: str, model_line: str) -> str:
    """Markdown table: one column per profile plus a total."""
    profiles = list(dict.fromkeys(r.profile for r in results))
    columns = profiles + ["all"]
    lines = [f"# Scorecard: {title}", "", f"{model_line}", "",
             "| Check | " + " | ".join(columns) + " |",
             "|---|" + "---|" * len(columns)]
    for label, kind, check in ROWS:
        cells = [_count([r for r in results if r.profile == p], kind, check) for p in profiles]
        cells.append(_count(results, kind, check))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    speed = [_average([r for r in results if r.profile == p], ("answerable", "unanswerable",
             "qualifier")) for p in profiles] + [_average(results, ("answerable",
             "unanswerable", "qualifier"))]
    lines.append("| Seconds per answer | " + " | ".join(speed) + " |")
    facts_speed = [_average([r for r in results if r.profile == p], ("facts",))
                   for p in profiles] + [_average(results, ("facts",))]
    lines.append("| Seconds per facts | " + " | ".join(facts_speed) + " |")

    failures = [r for r in results if not r.passed]
    lines += ["", f"## Failures ({len(failures)})", ""]
    for r in failures:
        answer = " ".join(r.answer.split())
        lines.append(f"- **{r.profile}, {r.kind}**: {r.question}  \n"
                     f"  Problem: {r.detail}  \n"
                     f"  Sections: {', '.join(r.sections)}  \n"
                     f"  Answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")
    return "\n".join(lines) + "\n"


def save_results(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
                    encoding="utf-8")
