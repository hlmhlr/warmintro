"""Load and clean a profile (the lab's "extract the LinkedIn profile" step).

Run: uv run python scripts/02_load_profile.py
     uv run python scripts/02_load_profile.py --profile data/profiles/me/alex_rivera.json
"""

import argparse
import json

from warmintro.profiles import CONTACTS_DIR, Profile, clean_profile_data, parse_profile

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--profile", default=str(CONTACTS_DIR / "amara_okafor.json"))
args = parser.parse_args()

with open(args.profile, encoding="utf-8") as file:
    raw = json.load(file)
cleaned = clean_profile_data(raw)
profile = parse_profile(raw)

print(f"Loaded {args.profile}")
print(f"Top-level fields in the file: {len(raw)}")
print("Removed while cleaning (empty, unwanted or contact details):",
      sorted(set(raw) - set(cleaned)) or "none")
print("Ignored (not part of the profile schema):",
      sorted(set(cleaned) - set(Profile.model_fields)) or "none")

print(f"\n{profile.full_name}: {profile.headline}")
print(f"Location: {profile.city}, {profile.country}")
for name in ("experiences", "education", "skills", "languages", "certifications",
             "volunteer_work", "projects", "awards", "interests"):
    print(f"  {name:<15} {len(getattr(profile, name))}")
