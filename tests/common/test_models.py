from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from common.models import CandidateProfile, Posting

EXAMPLE_PROFILE_PATH = "common/candidate_profile.example.json"


def test_candidate_profile_loads_from_file():
    profile = CandidateProfile.from_file(EXAMPLE_PROFILE_PATH)

    assert profile.candidate_id == "example-candidate"
    assert "Python" in profile.skills
    assert len(profile.experience) == 1
    assert len(profile.projects) == 1


def test_candidate_profile_is_pluggable_not_hardcoded():
    other = CandidateProfile(
        candidate_id="someone-else",
        resume_text="Data scientist.",
        skills=["R", "Statistics"],
    )

    assert other.candidate_id != "example-candidate"
    assert other.experience == []
    assert other.projects == []


def test_candidate_profile_section_texts_covers_all_sections():
    profile = CandidateProfile.from_file(EXAMPLE_PROFILE_PATH)
    sections = profile.section_texts()

    assert set(sections) == {"resume", "skills", "experience", "projects"}
    assert "Python" in sections["skills"]
    assert "Example Corp" in sections["experience"]
    assert "Example Project" in sections["projects"]


def test_posting_requires_core_identity_fields():
    with pytest.raises(ValidationError):
        Posting()  # missing posting_id, source, source_url, posting_hash, ingested_at


def test_posting_defaults_are_safe_for_an_unscored_new_record():
    posting = Posting(
        posting_id="abc123",
        source="remoteok",
        source_url="https://example.com/job/1",
        posting_hash="abc123",
        ingested_at=datetime.now(timezone.utc),
    )

    assert posting.required_skills == []
    assert posting.remote_status == "unknown"
    assert posting.visa_status == "unknown"
    assert posting.score is None
    assert posting.alert_sent is False


def test_posting_rejects_inverted_comp_range():
    with pytest.raises(ValidationError):
        Posting(
            posting_id="abc123",
            source="remoteok",
            source_url="https://example.com/job/1",
            posting_hash="abc123",
            ingested_at=datetime.now(timezone.utc),
            comp_min=200_000,
            comp_max=100_000,
        )
