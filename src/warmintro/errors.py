"""Turn exceptions into short messages for people, never tracebacks.

Used by the CLI and the web app. The lab showed f"Error: {e}" with internal
details in the UI.
"""

import logging

from warmintro.fallback import AllProvidersFailed
from warmintro.llm import MissingExtra

logger = logging.getLogger(__name__)

BUSY = ("The free models are busy right now (rate limit or high demand). "
        "Please try again in a few minutes.")


def _status(error: Exception) -> int | None:
    """The HTTP status of an API error: OpenAI-style .status_code or Gemini's .code."""
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    return None


def friendly_error(error: Exception) -> str:
    """A one-line message for the user. The details go to the log."""
    logger.error("Request failed: %s: %s", type(error).__name__, error)
    logger.debug("Traceback", exc_info=error)  # shown only with debug logging

    if isinstance(error, MissingExtra):
        return str(error)  # already names the install command
    if isinstance(error, AllProvidersFailed):
        if error.errors and all(_status(e) in (401, 403) for _, e in error.errors):
            return "The API keys were rejected. Check them in .env (or on Render)."
        return BUSY
    if type(error).__name__ in ("APIConnectionError", "ConnectError"):  # cloud APIs
        return "Can't reach the model's server. Check your internet connection and try again."
    if isinstance(error, ConnectionError):
        return "Can't reach the model. Is Ollama running? Start the Ollama app and try again."
    if type(error).__name__ in ("ReadTimeout", "TimeoutException", "TimeoutError",
                                "APITimeoutError"):
        return "The model took too long to answer. Please try again."

    if "MAX_TOKENS" in str(error):
        return "The answer was cut off at the length limit. Try a shorter question."

    code = _status(error)
    if code == 429:
        return "The free API limit was reached. Wait a minute and try again."
    if code in (500, 502, 503, 504):  # e.g. Gemini's 503 "high demand", 504 deadline
        return BUSY
    if code in (401, 403) or (code == 400 and "key" in str(error).lower()):
        return "The API key was rejected. Check it in .env."
    if isinstance(error, ValueError):
        return str(error)  # our own validation messages are written for users
    return "Something went wrong while answering. Please try again."
