"""Gradio web app (replaces the lab's app.py).

Fixes from the lab's version:
- Sessions: the lab kept every index in a global dict that only grew. Here
  each browser session holds its own state (gr.State with a time limit), and
  launch() caps the number of stored sessions.
- Model choice: the lab's dropdown changed a global setting for all users.
  Here the chosen model is stored in the user's own session.
- Errors: the lab showed f"Error: {e}". Here users get a short message and
  the details go to the server log.
- Data: no scraping; sample profiles, or a pasted profile with contact
  details removed.
"""

import logging
import os

import gradio as gr

from warmintro.common_ground import find_common_ground
from warmintro.config import ANSWER_GENERATION, FACTS_GENERATION, load_settings
from warmintro.errors import friendly_error
from warmintro.index import build_profile_index
from warmintro.llm import describe, describe_llm, get_embeddings, get_llm
from warmintro.profiles import (
    CONTACTS_DIR,
    ME_DIR,
    Profile,
    list_profiles,
    load_profile,
    parse_pasted_profile,
)
from warmintro.query import answer_user_query, generate_initial_facts

logger = logging.getLogger(__name__)

PASTE, NONE = "paste", "none"
AUTO = "auto"  # model choice for a provider chain: each provider uses its own model
SESSION_SECONDS = 60 * 60   # a session's index is dropped after an hour
MAX_SESSIONS = 50           # sessions kept in memory at once (free server: 512 MB)
EXAMPLE_QUESTIONS = [
    "What is this person's current job?",
    "Where did this person study?",
    "What does this person do outside of work?",
]


def _label(path) -> str:
    return load_profile(path).full_name


def contact_choices() -> list[tuple[str, str]]:
    samples = [(f"{_label(p)} (sample)", name) for name, p in list_profiles(CONTACTS_DIR).items()]
    return samples + [("Paste a profile...", PASTE)]


def me_choices() -> list[tuple[str, str]]:
    samples = [(f"{_label(p)} (sample)", name) for name, p in list_profiles(ME_DIR).items()]
    return [("No, skip this", NONE)] + samples + [("Paste my profile...", PASTE)]


def available_models() -> list[tuple[str, str]]:
    """Models the user can pick, as (label, value). Ollama: the installed ones."""
    settings = load_settings()
    if len(settings.llm_providers) > 1:
        return [(f"automatic: {describe_llm(settings)}", AUTO)]
    if settings.llm_provider != "ollama":
        name = describe_llm(settings).split(" (")[0]
        return [(name, name)]
    try:
        import ollama

        names = [m.model for m in ollama.Client(settings.ollama_base_url).list().models]
        chat_models = [name for name in names if "embed" not in name]
    except Exception:  # Ollama not running yet: offer the configured model
        chat_models = []
    if settings.ollama_model not in chat_models:
        chat_models = [settings.ollama_model] + chat_models
    return [(name, name) for name in chat_models]


def _llm(generation, model: str):
    return get_llm(generation, model=None if model == AUTO else model)


def _answered_by(llm) -> str:
    """For a provider chain: which provider wrote the answer."""
    provider = getattr(llm, "last_provider", None)
    return f"\n\n<sub>Written by {provider}</sub>" if provider else ""


def max_questions() -> int:
    """Questions allowed per session (0 = no limit). Protects a free API key."""
    return int(os.getenv("MAX_QUESTIONS_PER_SESSION", "0") or 0)


def _profile(choice: str, pasted: str, folder, default_name: str) -> Profile:
    if choice == PASTE:
        return parse_pasted_profile(pasted, default_name=default_name)
    samples = list_profiles(folder)
    if choice not in samples:
        raise ValueError("Please choose a profile.")
    return load_profile(samples[choice])


def prepare(contact: str, contact_text: str, me: str, me_text: str, model: str):
    """Build the index and write the facts (and common ground if `me` is set).

    Returns: facts markdown, common-ground markdown, session state, empty chat.
    """
    try:
        profile = _profile(contact, contact_text, CONTACTS_DIR, "This person")
        my_profile = None if me == NONE else _profile(me, me_text, ME_DIR, "You")
        index = build_profile_index({"them": profile})
        llm = _llm(FACTS_GENERATION, model)
        facts = generate_initial_facts(index, llm=llm)
        facts_by = _answered_by(llm)
        common = find_common_ground(my_profile, profile, llm=llm) if my_profile else None
        common_by = _answered_by(llm)
    except Exception as error:
        return f"⚠️ {friendly_error(error)}", "", None, []

    facts_md = f"### 3 facts about {profile.full_name}\n\n{facts.text}{facts_by}"
    common_md = (f"### What you have in common\n\n{common.text}{common_by}" if common else "")
    session = {"index": index, "name": profile.full_name, "model": model, "questions": 0}
    return facts_md, common_md, session, []


def ask(question: str, history: list[dict], session: dict | None):
    """Answer one question about the loaded profile. Returns: cleared box, chat."""
    question = (question or "").strip()
    history = list(history or [])
    if not question:
        return "", history
    history.append({"role": "user", "content": question})

    limit = max_questions()
    if session is None:
        reply = "Load a profile first (step 1 above)."
    elif limit and session["questions"] >= limit:
        reply = (f"This demo allows {limit} questions per profile. "
                 "Load the profile again to continue.")
    else:
        try:
            llm = _llm(ANSWER_GENERATION, session["model"])
            reply = answer_user_query(session["index"], question, llm=llm).text
            provider = getattr(llm, "last_provider", None)
            reply += f" *({provider})*" if provider else ""
            session["questions"] += 1
        except Exception as error:
            reply = f"⚠️ {friendly_error(error)}"
    history.append({"role": "assistant", "content": reply})
    return "", history


def show_paste_box(choice: str):
    return gr.update(visible=choice == PASTE)


def build_ui() -> gr.Blocks:
    models = available_models()
    with gr.Blocks(title="warmintro") as demo:
        gr.Markdown(
            "# warmintro\n"
            "Conversation starters from a professional profile: three facts with "
            "questions, what you have in common, and answers grounded in the profile.\n\n"
            "Sample profiles are fictional. If you paste a profile, use your own or "
            "fictional data; email addresses and phone numbers are removed.\n\n"
            f"<sub>{describe()}</sub>")

        with gr.Row():
            with gr.Column():
                contact = gr.Dropdown(contact_choices(), value="amara_okafor",
                                      label="1. Who do you want to meet?")
                contact_text = gr.Textbox(label="Their profile (JSON or plain text)",
                                          lines=6, visible=False)
            with gr.Column():
                me = gr.Dropdown(me_choices(), value=NONE,
                                 label="Optional: you, to find common ground")
                me_text = gr.Textbox(label="Your profile (JSON or plain text)",
                                     lines=6, visible=False)
        with gr.Row():
            model = gr.Dropdown(models, value=models[0][1], label="Model",
                                interactive=len(models) > 1)
            start = gr.Button("Get icebreakers", variant="primary")

        facts = gr.Markdown()
        common = gr.Markdown()
        session = gr.State(None, time_to_live=SESSION_SECONDS)

        gr.Markdown("### 2. Ask about this person")
        chatbot = gr.Chatbot(height=380, label="Answers come only from the profile")
        with gr.Row():
            question = gr.Textbox(placeholder="e.g. Where did this person study?",
                                  show_label=False, scale=5)
            send = gr.Button("Ask", scale=1)
        gr.Examples(EXAMPLE_QUESTIONS, inputs=question)

        contact.change(show_paste_box, contact, contact_text, api_name="toggle_contact")
        me.change(show_paste_box, me, me_text, api_name="toggle_me")
        start.click(prepare, [contact, contact_text, me, me_text, model],
                    [facts, common, session, chatbot], api_name="icebreakers")
        for event in (send.click, question.submit):
            event(ask, [question, chatbot, session], [question, chatbot], api_name="ask")
    return demo


def server_settings() -> dict:
    """Listen on all interfaces at $PORT when a host (Render) sets it, else locally."""
    port = os.getenv("PORT")
    if port:
        return {"server_name": "0.0.0.0", "server_port": int(port)}
    return {"server_name": "127.0.0.1", "server_port": 7860}


def check_models() -> None:
    """Create both models once, so a wrong setup stops the app at startup
    (e.g. LLM_PROVIDER=gemini without its packages) instead of failing on
    the first click. Creating them makes no network calls."""
    try:
        get_llm()
        get_embeddings()
    except Exception as error:
        raise SystemExit(f"Can't start: {friendly_error(error)}\nSettings: {describe()}") \
            from None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    check_models()
    build_ui().launch(
        **server_settings(),
        show_error=False,                    # never send tracebacks to the browser
        ssr_mode=False,                      # no extra Node server (512 MB on Render)
        state_session_capacity=MAX_SESSIONS,
        theme=gr.themes.Soft(),
    )
