import json

import pytest
from pydantic import ValidationError

from warmintro.profiles import (
    CONTACTS_DIR,
    MAX_PASTE_CHARS,
    ME_DIR,
    clean_profile_data,
    list_profiles,
    load_profile,
    parse_pasted_profile,
    parse_profile,
    scrub_contact_details,
)


def test_cleaning_removes_empty_values_at_every_level():
    raw = {"full_name": "A", "summary": "", "skills": [],
           "experiences": [{"title": "Dev", "company": "X", "description": ""}]}
    assert clean_profile_data(raw) == {
        "full_name": "A", "experiences": [{"title": "Dev", "company": "X"}]}


def test_cleaning_drops_unwanted_fields_and_contact_details():
    raw = {"full_name": "A", "people_also_viewed": ["B"], "profile_pic_url": "x",
           "personal_emails": ["a@example.com"], "phone": "123",
           "inferred_salary": {"min": 1}}
    assert clean_profile_data(raw) == {"full_name": "A"}


def test_cleaning_strips_whitespace():
    assert clean_profile_data({"headline": "  Hi  ", "city": "   "}) == {"headline": "Hi"}


def test_unknown_fields_are_ignored():
    profile = parse_profile({"full_name": "A", "follower_count": 10})
    assert not hasattr(profile, "follower_count")


def test_a_profile_needs_a_name():
    with pytest.raises(ValidationError):
        parse_profile({"headline": "No name"})
    with pytest.raises(ValidationError):
        parse_profile({"full_name": "   "})  # only spaces is empty after cleaning


def test_current_job_has_no_end_date():
    profile = load_profile(CONTACTS_DIR / "amara_okafor.json")
    assert profile.experiences[0].ends_at is None
    # The empty description of the third job was cleaned away.
    assert profile.experiences[2].description is None


@pytest.mark.parametrize("folder, count", [(CONTACTS_DIR, 3), (ME_DIR, 2)])
def test_every_sample_profile_loads(folder, count):
    files = list_profiles(folder)
    assert len(files) == count
    for path in files.values():
        profile = load_profile(path)
        assert profile.full_name and profile.experiences and profile.education


def test_no_sample_profile_keeps_contact_details():
    for folder in (CONTACTS_DIR, ME_DIR):
        for path in list_profiles(folder).values():
            dumped = json.dumps(load_profile(path).model_dump())
            assert "@" not in dumped


@pytest.mark.parametrize("text, expected", [
    ("Mail me at kim.lee@example.com today", "Mail me at [removed] today"),
    ("Call +39 333 123 4567 now", "Call [removed] now"),
    ("From 2017-09 to 2021-03, 40,000 users, 15 fps", "From 2017-09 to 2021-03, 40,000 users, 15 fps"),
])
def test_contact_details_inside_text_are_scrubbed(text, expected):
    assert scrub_contact_details(text) == expected


def test_scrubbing_reaches_nested_text():
    profile = parse_profile({"full_name": "A", "summary": "Write to a@example.com",
                             "experiences": [{"title": "Dev", "company": "X",
                                              "description": "Phone 0612345678 ok"}]})
    assert profile.summary == "Write to [removed]"
    assert profile.experiences[0].description == "Phone [removed] ok"


def test_pasted_json_profile():
    profile = parse_pasted_profile('{"full_name": "Kim", "skills": ["Go"]}')
    assert (profile.full_name, profile.skills) == ("Kim", ["Go"])


def test_pasted_plain_text_becomes_a_summary():
    profile = parse_pasted_profile("Kim builds compilers in Oslo. kim@example.com",
                                   default_name="This person")
    assert profile.full_name == "This person"
    assert profile.summary == "Kim builds compilers in Oslo. [removed]"


@pytest.mark.parametrize("text, message", [
    ("   ", "Paste a profile first"),
    ("x" * (MAX_PASTE_CHARS + 1), "too long"),
    ('{"full_name": "Kim",', "could not be read"),
    ('{"headline": "No name"}', "full_name"),
])
def test_bad_pastes_get_a_clear_message(text, message):
    with pytest.raises(ValueError, match=message):
        parse_pasted_profile(text)
