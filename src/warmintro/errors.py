"""Turn exceptions into short messages for people, never tracebacks.

Used by the CLI and the web app. The lab showed f"Error: {e}" with internal
details in the UI.
"""

import logging

from warmintro.llm import MissingExtra

logger = logging.getLogger(__name__)


def friendly_error(error: Exception) -> str:
    """A one-line message for the user. The details go to the log."""
    logger.error("Request failed: %s: %s", type(error).__name__, error)
    logger.debug("Traceback", exc_info=error)  # shown only with debug logging

    if isinstance(error, MissingExtra):
        return str(error)  # already names the install command
    if isinstance(error, ConnectionError):
        return "Can't reach the model. Is Ollama running? Start the Ollama app and try again."
    if type(error).__name__ in ("ReadTimeout", "TimeoutException", "TimeoutError"):
        return "The model took too long to answer. Please try again."

    code = getattr(error, "code", None)  # Gemini API errors carry an HTTP code
    if code == 429:
        return "The free API limit was reached. Wait a minute and try again."
    if code in (400, 401, 403) and "key" in str(error).lower():
        return "The API key was rejected. Check GOOGLE_API_KEY."
    if isinstance(error, ValueError):
        return str(error)  # our own validation messages are written for users
    return "Something went wrong while answering. Please try again."
