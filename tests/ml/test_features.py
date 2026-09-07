from datetime import date, datetime, timezone

from common.models import CandidateProfile, ExperienceEntry, Posting
from ml.features import (
    FEATURE_NAMES,
    build_feature_vector,
    comp_range_fit,
    estimate_candidate_seniority,
    feature_vector_to_array,
    remote_match,
    seniority_match,
    skill_overlap_count,
    visa_match,
)


def _posting(**overrides) -> Posting:
    defaults = {
        "posting_id": "p1",
        "source": "remoteok",
        "source_url": "https://example.com/job/1",
        "posting_hash": "p1",
        "ingested_at": datetime.now(timezone.utc),
        "required_skills": ["Python", "AWS"],
        "seniority": "senior",
        "remote_status": "remote",
        "visa_status": "sponsors",
        "comp_min": 100_000,
        "comp_max": 150_000,
        "embedding_similarity": 0.8,
    }
    defaults.update(overrides)
    return Posting(**defaults)


def _candidate(**overrides) -> CandidateProfile:
    defaults = {
        "candidate_id": "c1",
        "resume_text": "Experienced engineer.",
        "skills": ["python", "aws", "sql"],
        "desired_comp_min": 120_000,
        "remote_preference": "remote",
        "needs_sponsorship": False,
    }
    defaults.update(overrides)
    return CandidateProfile(**defaults)


# --- skill_overlap_count ---


def test_skill_overlap_count_is_case_insensitive():
    posting = _posting(required_skills=["Python", "AWS", "Kubernetes"])
    candidate = _candidate(skills=["python", "aws", "sql"])

    assert skill_overlap_count(posting, candidate) == 2.0


def test_skill_overlap_count_is_zero_for_no_shared_skills():
    posting = _posting(required_skills=["Rust"])
    candidate = _candidate(skills=["python"])

    assert skill_overlap_count(posting, candidate) == 0.0


# --- estimate_candidate_seniority ---


def test_estimate_seniority_returns_none_with_no_experience():
    candidate = _candidate(experience=[])

    assert estimate_candidate_seniority(candidate) is None


def test_estimate_seniority_junior_under_two_years():
    candidate = _candidate(
        experience=[
            ExperienceEntry(title="Eng", company="X", start_date="2025-01", end_date="2026-01")
        ]
    )

    assert estimate_candidate_seniority(candidate, as_of=date(2026, 9, 1)) == "junior"


def test_estimate_seniority_senior_around_six_years():
    candidate = _candidate(
        experience=[
            ExperienceEntry(title="Eng", company="X", start_date="2020-01", end_date=None)
        ]
    )

    assert estimate_candidate_seniority(candidate, as_of=date(2026, 1, 1)) == "senior"


def test_estimate_seniority_sums_multiple_entries():
    candidate = _candidate(
        experience=[
            ExperienceEntry(title="Eng", company="X", start_date="2015-01", end_date="2020-01"),
            ExperienceEntry(title="Eng", company="Y", start_date="2020-01", end_date="2026-01"),
        ]
    )

    # 5 years + 6 years = 11 years -> staff (8-12 band)
    assert estimate_candidate_seniority(candidate, as_of=date(2026, 1, 1)) == "staff"


# --- seniority_match ---


def test_seniority_match_is_one_for_exact_match():
    posting = _posting(seniority="senior")
    candidate = _candidate(
        experience=[
            ExperienceEntry(title="Eng", company="X", start_date="2020-01", end_date=None)
        ]
    )

    assert seniority_match(posting, candidate, as_of=date(2026, 1, 1)) == 1.0


def test_seniority_match_decays_with_distance():
    senior_posting = _posting(seniority="senior")  # 1 level from staff
    junior_posting = _posting(seniority="junior")  # 3 levels from staff
    candidate = _candidate(
        experience=[
            ExperienceEntry(title="Eng", company="X", start_date="2015-01", end_date=None)
        ]
    )  # ~11 years as of 2026-01 -> staff

    close = seniority_match(senior_posting, candidate, as_of=date(2026, 1, 1))
    far = seniority_match(junior_posting, candidate, as_of=date(2026, 1, 1))

    assert 0.0 <= far < close < 1.0


def test_seniority_match_is_neutral_when_posting_seniority_unmapped():
    posting = _posting(seniority=None)
    candidate = _candidate(
        experience=[
            ExperienceEntry(title="Eng", company="X", start_date="2020-01", end_date=None)
        ]
    )

    assert seniority_match(posting, candidate, as_of=date(2026, 1, 1)) == 0.5


def test_seniority_match_is_neutral_with_no_candidate_experience():
    posting = _posting(seniority="senior")
    candidate = _candidate(experience=[])

    assert seniority_match(posting, candidate) == 0.5


# --- comp_range_fit ---


def test_comp_range_fit_is_neutral_with_no_stated_desired_comp():
    posting = _posting(comp_min=100_000, comp_max=150_000)
    candidate = _candidate(desired_comp_min=None)

    assert comp_range_fit(posting, candidate) == 0.5


def test_comp_range_fit_is_neutral_when_posting_discloses_no_comp():
    posting = _posting(comp_min=None, comp_max=None)
    candidate = _candidate(desired_comp_min=120_000)

    assert comp_range_fit(posting, candidate) == 0.5


def test_comp_range_fit_is_one_when_posting_meets_desired_minimum():
    posting = _posting(comp_min=100_000, comp_max=150_000)
    candidate = _candidate(desired_comp_min=120_000)

    assert comp_range_fit(posting, candidate) == 1.0


def test_comp_range_fit_is_partial_credit_when_posting_falls_short():
    posting = _posting(comp_min=50_000, comp_max=60_000)
    candidate = _candidate(desired_comp_min=120_000)

    assert comp_range_fit(posting, candidate) == 60_000 / 120_000


# --- remote_match ---


def test_remote_match_is_one_for_exact_match():
    posting = _posting(remote_status="remote")
    candidate = _candidate(remote_preference="remote")

    assert remote_match(posting, candidate) == 1.0


def test_remote_match_is_zero_for_mismatch():
    posting = _posting(remote_status="onsite")
    candidate = _candidate(remote_preference="remote")

    assert remote_match(posting, candidate) == 0.0


def test_remote_match_is_neutral_with_no_candidate_preference():
    posting = _posting(remote_status="onsite")
    candidate = _candidate(remote_preference="unknown")

    assert remote_match(posting, candidate) == 0.5


def test_remote_match_is_neutral_when_posting_undisclosed():
    posting = _posting(remote_status="unknown")
    candidate = _candidate(remote_preference="remote")

    assert remote_match(posting, candidate) == 0.5


# --- visa_match ---


def test_visa_match_is_one_when_candidate_does_not_need_sponsorship():
    posting = _posting(visa_status="no_sponsorship")
    candidate = _candidate(needs_sponsorship=False)

    assert visa_match(posting, candidate) == 1.0


def test_visa_match_is_one_when_posting_sponsors_and_candidate_needs_it():
    posting = _posting(visa_status="sponsors")
    candidate = _candidate(needs_sponsorship=True)

    assert visa_match(posting, candidate) == 1.0


def test_visa_match_is_zero_when_posting_does_not_sponsor_and_candidate_needs_it():
    posting = _posting(visa_status="no_sponsorship")
    candidate = _candidate(needs_sponsorship=True)

    assert visa_match(posting, candidate) == 0.0


def test_visa_match_is_neutral_when_posting_undisclosed_and_candidate_needs_it():
    posting = _posting(visa_status="unknown")
    candidate = _candidate(needs_sponsorship=True)

    assert visa_match(posting, candidate) == 0.5


# --- build_feature_vector / feature_vector_to_array ---


def test_build_feature_vector_has_all_expected_keys():
    posting = _posting()
    candidate = _candidate()

    vector = build_feature_vector(posting, candidate, as_of=date(2026, 1, 1))

    assert set(vector) == set(FEATURE_NAMES)


def test_build_feature_vector_defaults_missing_embedding_similarity_to_zero():
    posting = _posting(embedding_similarity=None)
    candidate = _candidate()

    vector = build_feature_vector(posting, candidate)

    assert vector["embedding_similarity"] == 0.0


def test_feature_vector_to_array_preserves_feature_name_order():
    posting = _posting()
    candidate = _candidate()

    vector = build_feature_vector(posting, candidate, as_of=date(2026, 1, 1))
    array = feature_vector_to_array(vector)

    assert array == [vector[name] for name in FEATURE_NAMES]
    assert len(array) == len(FEATURE_NAMES)
