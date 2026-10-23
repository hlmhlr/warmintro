import pytest

from warmintro.errors import friendly_error
from warmintro.llm import MissingExtra


class ApiError(Exception):
    """Shaped like google.genai's errors: an HTTP code and a message."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class ReadTimeout(Exception):
    """Same name as httpx's timeout error."""


@pytest.mark.parametrize("error, expected", [
    (ConnectionError("Failed to connect to Ollama"), "Is Ollama running?"),
    (ReadTimeout("slow"), "took too long"),
    (ApiError(429, "RESOURCE_EXHAUSTED"), "free API limit"),
    (ApiError(400, "API key not valid"), "GOOGLE_API_KEY"),
    (ValueError("The question is empty."), "The question is empty."),
    (MissingExtra("LLM_PROVIDER=gemini", "gemini", ImportError(name="google")),
     "uv sync --extra gemini"),
])
def test_known_errors_get_a_helpful_message(error, expected):
    assert expected in friendly_error(error)


def test_unknown_errors_hide_internal_details():
    message = friendly_error(RuntimeError("secret internal path /srv/app.py line 3"))
    assert "secret" not in message and "Something went wrong" in message
