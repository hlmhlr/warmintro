"""Load, clean and validate professional profiles.

Replaces the lab's data_extraction.py. The lab fetched LinkedIn profiles of
real people through Proxycurl (now discontinued). Here profiles are local
JSON files: fictional samples written for this repo, or a profile the user
pastes in. Contact details are always dropped.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = ROOT / "data" / "profiles"
CONTACTS_DIR = PROFILES_DIR / "contacts"  # people you want to meet
ME_DIR = PROFILES_DIR / "me"              # your own profile

# Fields the lab removed because they add noise, plus contact details,
# which an icebreaker never needs.
UNWANTED_KEYS = {"people_also_viewed", "similarly_named_profiles",
                 "profile_pic_url", "background_cover_image_url"}
CONTACT_KEYS = {"personal_emails", "personal_numbers", "email", "phone",
                "inferred_salary"}


class _Part(BaseModel):
    # Unknown keys are ignored, so a profile with extra fields still loads.
    model_config = ConfigDict(extra="ignore")


class Experience(_Part):
    title: str
    company: str
    location: str | None = None
    starts_at: str | None = None   # "2021" or "2021-03"
    ends_at: str | None = None     # None = current job
    description: str | None = None


class Education(_Part):
    school: str
    degree_name: str | None = None
    field_of_study: str | None = None
    starts_at: str | None = None
    ends_at: str | None = None
    description: str | None = None


class Certification(_Part):
    name: str
    authority: str | None = None
    year: str | None = None


class Volunteering(_Part):
    role: str
    organization: str
    cause: str | None = None
    description: str | None = None


class Project(_Part):
    title: str
    description: str | None = None


class Award(_Part):
    title: str
    issuer: str | None = None
    year: str | None = None


class Profile(_Part):
    full_name: str = Field(min_length=1)
    headline: str | None = None
    city: str | None = None
    country: str | None = None
    summary: str | None = None
    experiences: list[Experience] = []
    education: list[Education] = []
    skills: list[str] = []
    languages: list[str] = []
    certifications: list[Certification] = []
    volunteer_work: list[Volunteering] = []
    projects: list[Project] = []
    awards: list[Award] = []
    interests: list[str] = []


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {})


def clean_profile_data(raw: Any) -> Any:
    """Remove empty values and unwanted fields, at every level.

    The lab cleaned only the top level, so empty fields inside jobs or
    schools stayed in the text sent to the model.
    """
    if isinstance(raw, dict):
        cleaned = {}
        for key, value in raw.items():
            if key in UNWANTED_KEYS or key in CONTACT_KEYS:
                continue
            value = clean_profile_data(value)
            if not _is_empty(value):
                cleaned[key] = value
        return cleaned
    if isinstance(raw, list):
        items = [clean_profile_data(item) for item in raw]
        return [item for item in items if not _is_empty(item)]
    if isinstance(raw, str):
        return raw.strip()
    return raw


def parse_profile(raw: dict) -> Profile:
    """Clean a profile dictionary and check it has the expected shape."""
    return Profile.model_validate(clean_profile_data(raw))


def load_profile(path: str | Path) -> Profile:
    """Read a profile JSON file."""
    with open(path, encoding="utf-8") as file:
        return parse_profile(json.load(file))


def list_profiles(folder: Path = CONTACTS_DIR) -> dict[str, Path]:
    """Profile files in a folder, keyed by file name without .json."""
    return {path.stem: path for path in sorted(folder.glob("*.json"))}
